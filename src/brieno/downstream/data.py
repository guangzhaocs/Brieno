"""Chunk-level datasets for the classification tasks.

A dataset is a tree of per-run chunk files plus index tables that define the splits::

    <root>/meta/<class>/<run>/chunks/chunks_part0000.npy   (N, 1, 16384) float16
                             chunks/chunk_index.npy        (N, 3) int32
    <root>/<setting>/fold<k>/{train,val,test}.tsv          one row per run

Splits are stored as **indices**, never as copies of the signal: a fold is a list of
which chunks of which runs belong to it, so the same chunk files serve every fold and
a new split costs a table rather than a dataset.

`chunk_index.npy` rows are `(read_row, raw_start, n_chunks_kept)`, and a chunk's global
number `k` inside a run locates it as part `k // 10000`, row `k % 10000`.
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

__all__ = ["PART", "load_split", "subsample", "ChunkDataset", "PartIterDataset",
           "BalancedSampler"]

PART = 10000          # chunks per part file, fixed by the builder


def _resolve(path, root):
    """Make a path from the index table usable under ``root``.

    Index tables are written with the absolute paths of the machine that built them.
    Rather than demanding they be rewritten, re-root anything that no longer exists on
    the ``meta/`` component, which is the one fixed point of the layout.
    """
    if os.path.exists(path):
        return path
    parts = path.replace("\\", "/").split("/")
    if "meta" in parts:
        return os.path.join(root, *parts[parts.index("meta"):])
    raise FileNotFoundError(
        f"{path} does not exist and cannot be re-rooted at {root}: the index table "
        "refers to a layout without a 'meta' directory. Rebuild the index for this "
        "dataset.")


def load_split(tsv, root=None):
    """Read one split table into flat arrays over all of its runs.

    Returns a dict with ``runs`` (the table rows) and, per chunk: ``run_idx``,
    ``chunk_k``, ``read_gid``, ``label`` and ``pos``.

    ``read_gid`` packs the run index and the read row into one integer so that chunks
    of the same read stay groupable after the per-run tables are concatenated — read
    level results need that grouping, and a plain row number would collide across runs.
    """
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(tsv))))
    runs = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))

    run_idx, chunk_k, read_gid, label, pos = [], [], [], [], []
    for i, r in enumerate(runs):
        r["chunks_dir"] = _resolve(r["chunks_dir"], root)
        r["index_npy"] = _resolve(r["index_npy"], root)
        ci = np.load(os.path.join(r["chunks_dir"], "chunk_index.npy"))
        rows = np.load(r["index_npy"])                     # read rows in this split
        k = np.nonzero(np.isin(ci[:, 0], rows))[0].astype(np.int32)

        run_idx.append(np.full(len(k), i, np.int16))
        chunk_k.append(k)
        read_gid.append((i << 24) + ci[k, 0].astype(np.int64))
        label.append(np.full(len(k), int(r["label"]), np.int8))
        pos.append(ci[k, 1].astype(np.int64))
        assert len(k) == int(r["n_chunks"]), (r["run"], len(k), r["n_chunks"])

    return dict(runs=runs,
                run_idx=np.concatenate(run_idx), chunk_k=np.concatenate(chunk_k),
                read_gid=np.concatenate(read_gid), label=np.concatenate(label),
                pos=np.concatenate(pos))


def subsample(split, n, seed):
    """A uniform random subset of a split, kept in order, for fast per-epoch validation."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(split["chunk_k"]), min(n, len(split["chunk_k"])),
                             replace=False))
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in split.items()}


class ChunkDataset(Dataset):
    """Random access to chunks through per-(run, part) memory maps, opened per worker."""

    def __init__(self, split):
        self.s = split
        self.dirs = [r["chunks_dir"] for r in split["runs"]]
        self._mm = {}

    def __len__(self):
        return len(self.s["chunk_k"])

    def _part(self, ri, p):
        key = (ri, p)
        if key not in self._mm:
            self._mm[key] = np.load(os.path.join(self.dirs[ri], f"chunks_part{p:04d}.npy"),
                                    mmap_mode="r")
        return self._mm[key]

    def __getitem__(self, i):
        ri, k = int(self.s["run_idx"][i]), int(self.s["chunk_k"][i])
        row = self._part(ri, k // PART)[k % PART]
        return (torch.from_numpy(np.asarray(row, dtype=np.float32)),
                int(self.s["label"][i]), int(self.s["read_gid"][i]),
                int(self.s["pos"][i]), i)


class PartIterDataset(IterableDataset):
    """A sequential pass over a split, one whole part file at a time.

    Evaluation touches most of a split, and reading whole parts is far faster than
    touching rows through a memory map — the rows a split wants are scattered, so
    row-wise access pays a seek each time. Workers take every n-th (run, part) group.
    Requires the split to be ordered by (run, chunk_k), as ``load_split`` produces.
    """

    def __init__(self, split):
        self.s = split
        self.dirs = [r["chunks_dir"] for r in split["runs"]]

    def __iter__(self):
        ri, ck = self.s["run_idx"], self.s["chunk_k"]
        key = ri.astype(np.int64) * 100000 + ck // PART
        bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
        groups = list(zip(bounds[:-1], bounds[1:]))

        info = torch.utils.data.get_worker_info()
        if info is not None:
            groups = groups[info.id::info.num_workers]

        for b0, b1 in groups:
            r, p = int(ri[b0]), int(ck[b0] // PART)
            arr = np.load(os.path.join(self.dirs[r], f"chunks_part{p:04d}.npy"))
            x = arr[ck[b0:b1] % PART].astype(np.float32)
            del arr
            for j in range(b1 - b0):
                i = b0 + j
                yield (torch.from_numpy(x[j]), int(self.s["label"][i]),
                       int(self.s["read_gid"][i]), int(self.s["pos"][i]), i)


class BalancedSampler(Sampler):
    """Each epoch draws the same number of chunks from every class.

    Class sizes in these datasets differ by an order of magnitude — a run yields as
    many chunks as it has reads — and training on the natural distribution would let
    the largest class dominate the loss. Sampling with replacement where a class is
    short keeps the epoch length fixed and the classes equal.

    The seed depends on the epoch, so an interrupted run that resumes at epoch N draws
    epoch N's chunks rather than replaying epoch 0.
    """

    def __init__(self, labels, n_per_class, n_classes, seed):
        self.by = [np.nonzero(labels == c)[0] for c in range(n_classes)]
        self.n = n_per_class
        self.seed = seed
        self.epoch = 0
        for c, b in enumerate(self.by):
            assert len(b) > 0, f"class {c} has no chunks in this split"

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        idx = np.concatenate([rng.choice(b, self.n, replace=len(b) < self.n)
                              for b in self.by])
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return self.n * len(self.by)
