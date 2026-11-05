"""Merge the per-file chunk shards of one dataset into a single array.

``brieno train prep`` writes one or more shards per input file. Tokenizer pretraining
instead reads one array per dataset, because the training sampler needs a single index
space per dataset to draw from and a stable train/validation split across runs.

The merge is done in two passes — measure, then fill a memory-mapped output — so a
corpus far larger than RAM merges on a laptop.
"""
from pathlib import Path

import numpy as np

__all__ = ["find_shards", "merge_shards"]


def find_shards(input_dir, recursive=False):
    """Sorted list of ``.npy`` shards under ``input_dir``."""
    input_dir = Path(input_dir)
    it = input_dir.rglob("*.npy") if recursive else input_dir.glob("*.npy")
    return sorted(p for p in it if p.is_file())


def merge_shards(input_dir, output_path, *, recursive=False, force=False, progress=True):
    """Concatenate every shard under ``input_dir`` into ``output_path``.

    All shards must agree on dtype and on everything but the first axis; a mismatch
    fails before any output is written, naming the offending file.

    Returns ``(n_chunks, n_shards)``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; use --force to overwrite")

    shards = find_shards(input_dir, recursive=recursive)
    if not shards:
        raise FileNotFoundError(f"no .npy shards under {input_dir}")

    # ---- pass 1: measure and validate, without reading any data ----
    total, dtype, tail = 0, None, None
    for shard in shards:
        arr = np.load(shard, mmap_mode="r")
        if dtype is None:
            dtype, tail = arr.dtype, arr.shape[1:]
        elif arr.dtype != dtype:
            raise ValueError(f"{shard.name}: dtype {arr.dtype}, expected {dtype}")
        elif arr.shape[1:] != tail:
            raise ValueError(f"{shard.name}: chunk shape {arr.shape[1:]}, expected {tail}")
        total += arr.shape[0]
        del arr

    print(f" * {len(shards)} shard(s) under {input_dir}")
    print(f" * {total:,} chunks of shape {tail}, {dtype} "
          f"-> {total * int(np.prod(tail)) * dtype.itemsize / 1e9:.2f} GB")

    # ---- pass 2: fill a memory-mapped output ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype,
                                    shape=(total, *tail))

    iterator = shards
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(shards, desc=f"  {output_path.name}", leave=False)
        except ImportError:
            pass

    at = 0
    for shard in iterator:
        arr = np.load(shard, mmap_mode="r")
        out[at:at + arr.shape[0]] = arr
        at += arr.shape[0]
        del arr
    out.flush()
    del out

    assert at == total, f"wrote {at} chunks, expected {total}"
    print(f" * wrote {output_path}")
    return total, len(shards)
