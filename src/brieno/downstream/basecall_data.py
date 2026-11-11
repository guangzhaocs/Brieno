"""Signal chunks paired with the bases they contain: the basecalling training data.

A run is a directory of shards plus a manifest::

    <root>/<dataset>/<run>/manifest.json
                           chunks_0000.signal.npy       (N, 1, chunk_size) float16
                           chunks_0000.refs_label.npy   (N, max_len) int8, 0 = pad, 1..4 = ACGT
                           chunks_0000.refs_len.npy     (N,) int32
                           chunks_0001. ...

``manifest.json`` lists the shards under ``"files"``; each entry names its
``signal_file``, ``refs_label_file`` and ``refs_len_file`` and gives ``n_chunks``.
A dataset is a directory of such runs.

The label of a chunk is the stretch of *reference* its signal aligns to, not the output
of another basecaller: training against a basecaller's own calls would teach its
errors. Chunks whose reference contains anything but A, C, G, T are expected to have
been dropped when the shards were built.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from brieno.models.vocab import ID_TO_BASE

__all__ = ["AlignedChunks", "AlignedCorpus", "open_chunks", "fixed_subset",
           "BlendedChunks", "BasecallCollator"]


class AlignedChunks(Dataset):
    """One run: ``__getitem__`` returns ``(signal [1, S] float32, ref_ids [L] int64)``.

    Shards are memory-mapped and opened lazily in each worker. Only the reference
    lengths are read up front, which is what the length filter needs.

    ``max_ref_len`` drops chunks with more bases than the head has room for: the CRF
    needs at least one output frame per base, and the released models emit 4096 frames
    per 16384-sample chunk. The filter is applied here, at construction, so ``len()``
    is the number of usable chunks.
    """

    def __init__(self, run_dir, max_ref_len=2047, min_ref_len=None):
        self.run_dir = str(run_dir)
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"manifest.json not found in {run_dir}")
        with open(manifest_path, encoding="utf-8") as fh:
            self.shards = json.load(fh)["files"]

        self._ref_len, shard_idx, chunk_idx = [], [], []
        n_total = 0
        for i, shard in enumerate(self.shards):
            lens = np.load(os.path.join(self.run_dir, shard["refs_len_file"]))
            if len(lens) != shard["n_chunks"]:
                raise ValueError(f"{self.run_dir}/{shard['refs_len_file']}: {len(lens)} "
                                 f"lengths, manifest says {shard['n_chunks']} chunks")
            keep = np.ones(len(lens), dtype=bool)
            if max_ref_len is not None:
                keep &= lens <= max_ref_len
            if min_ref_len is not None:
                keep &= lens >= min_ref_len
            kept = np.flatnonzero(keep)
            self._ref_len.append(lens)
            shard_idx.append(np.full(len(kept), i, dtype=np.int32))
            chunk_idx.append(kept.astype(np.int32))
            n_total += len(lens)

        # Two flat integer arrays rather than a list of tuples: a corpus of tens of
        # millions of chunks is indexed in a few hundred megabytes instead of several
        # gigabytes, and is not duplicated by copy-on-write in every worker.
        self._shard = np.concatenate(shard_idx) if shard_idx else np.zeros(0, np.int32)
        self._chunk = np.concatenate(chunk_idx) if chunk_idx else np.zeros(0, np.int32)
        self.n_before_filter = n_total
        self._signal, self._label = {}, {}                # per-worker memory maps

    def __len__(self):
        return len(self._shard)

    def __getitem__(self, idx):
        s, c = int(self._shard[idx]), int(self._chunk[idx])
        if s not in self._signal:
            shard = self.shards[s]
            self._signal[s] = np.load(os.path.join(self.run_dir, shard["signal_file"]),
                                      mmap_mode="r")
            self._label[s] = np.load(os.path.join(self.run_dir, shard["refs_label_file"]),
                                     mmap_mode="r")
        n = int(self._ref_len[s][c])
        signal = torch.from_numpy(self._signal[s][c].astype(np.float32).copy())   # [1, S]
        ref_ids = torch.from_numpy(self._label[s][c][:n].astype(np.int64).copy())  # [L]
        return signal, ref_ids


class AlignedCorpus(ConcatDataset):
    """Every run under one dataset directory, in sorted order."""

    def __init__(self, dataset_dir, **run_kwargs):
        dataset_dir = str(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
        runs = sorted(
            os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, d, "manifest.json")))
        if not runs:
            raise ValueError(f"no run with a manifest.json under {dataset_dir}")
        super().__init__([AlignedChunks(r, **run_kwargs) for r in runs])
        self.name = os.path.basename(dataset_dir.rstrip("/"))


def open_chunks(path, **run_kwargs):
    """A run directory or a dataset directory, whichever ``path`` is."""
    if os.path.isfile(os.path.join(path, "manifest.json")):
        return AlignedChunks(path, **run_kwargs)
    return AlignedCorpus(path, **run_kwargs)


def fixed_subset(dataset, n, seed):
    """The same ``n`` chunks every time: validation and test sets must not move between
    epochs, runs or machines, or their scores stop being comparable."""
    if not n or n >= len(dataset):
        return dataset
    idx = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    return Subset(dataset, sorted(idx.tolist()))


class BlendedChunks(Dataset):
    """Weighted mixture of datasets, sampled with replacement.

    Every item picks a dataset by weight and then a random chunk inside it, so the
    training mix is set by ``weights`` and not by how many chunks each dataset happens
    to have. One "epoch" is ``n_samples`` draws.
    """

    def __init__(self, datasets, weights, n_samples):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.prob_tensor = torch.tensor(weights, dtype=torch.float)
        self.n_samples = int(n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ds_idx = int(torch.multinomial(self.prob_tensor, 1).item())
        item = int(torch.randint(0, len(self.datasets[ds_idx]), (1,)).item())
        return self.datasets[ds_idx][item]


class BasecallCollator:
    """Batch of ``(signal, ref_ids)`` -> the tensors the CRF loss wants.

    Returns a dict with ``inputs`` ``[B, 1, S]``, ``targets`` (all reference ids
    concatenated), ``target_lengths`` ``[B]`` and ``ref_strs``, the references as
    strings for scoring decoded output against.
    """

    def __call__(self, batch):
        signals, labels = zip(*batch)
        return {
            "inputs": torch.stack(signals, dim=0),
            "targets": torch.cat(list(labels), dim=0),
            "target_lengths": torch.tensor([len(lab) for lab in labels], dtype=torch.long),
            "ref_strs": ["".join(ID_TO_BASE[int(i)] for i in lab) for lab in labels],
        }
