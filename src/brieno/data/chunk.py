"""Turn raw signal files into the fixed-length chunks the tokenizer pretrains on.

Per read: median/MAD normalise the whole read, trim the open-pore / stall region at the
start, then cut the remainder into overlapping windows. Chunks are written as ``float16``
``.npy`` shards of shape ``(N, 1, chunk_size)``, at most ``max_chunks_per_file`` per shard.
"""
import os
from pathlib import Path

import numpy as np
import torch

from brieno.data.signal import (
    SIGNAL_SUFFIXES,
    get_reader,
    med_mad_normalisation,
    trim,
)

__all__ = [
    "trim_threshold_for",
    "short_pad",
    "chunk_for_inference",
    "split_to_chunk",
    "find_signal_files",
    "chunk_one_file",
    "build_pretrain_chunks",
]

#: Default number of chunks per output shard.
MAX_CHUNKS_PER_FILE = 100_000

_TRIM_THRESHOLDS = {
    "dna_r9": 2.4,
    "dna_r10": 2.4,
    "rna_002": 1.5,
    "rna_004": 1.5,
}


def trim_threshold_for(modality: str) -> float:
    """Trim threshold (in normalised units) for a sequencing modality.

    DNA uses bonito's 2.4. The RNA value is provisional and has not been validated.
    """
    return _TRIM_THRESHOLDS.get(modality, 2.0)


def split_to_chunk(clean_normalized_signal, chunksize=10000, overlap=1000):
    """Cut one normalised, trimmed read into overlapping chunks.

    Returns a ``(N, 1, chunksize)`` half-precision tensor, or ``None`` when the read is
    shorter than one chunk. The read is aligned to its **end**: the remainder that does
    not fill a whole step is dropped from the start, so the final chunk ends exactly at
    the last sample (bonito's convention).
    """
    L = len(clean_normalized_signal)
    if L < chunksize:
        return None

    step = chunksize - overlap
    _, offset = divmod(L - chunksize, step)

    sig_tensor = torch.from_numpy(clean_normalized_signal[offset:]).half()
    blocks = sig_tensor.unfold(0, chunksize, step).unsqueeze(1)      # (N, 1, chunksize)
    return blocks


def find_signal_files(raw_dir, recursive: bool = False):
    """Sorted list of ``.fast5`` / ``.pod5`` files under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    it = raw_dir.rglob("*") if recursive else raw_dir.glob("*")
    return sorted(p for p in it if p.suffix in SIGNAL_SUFFIXES and p.is_file())


def chunk_one_file(path, out_dir, *, modality="dna_r9", min_length=16384,
                   chunk_size=16384, overlap=2048,
                   max_chunks_per_file=MAX_CHUNKS_PER_FILE, progress=True):
    """Chunk a single raw-signal file into ``out_dir``.

    Returns ``(n_shards, n_chunks, n_reads_failed)``.
    """
    threshold = trim_threshold_for(modality)
    stem = Path(path).stem
    out_dir = Path(out_dir)

    buffered, n_buffered, shard_index, n_chunks, n_failed = [], 0, 0, 0, 0

    def flush():
        nonlocal buffered, n_buffered, shard_index
        if not buffered:
            return
        tensor = torch.cat(buffered, dim=0)
        np.save(out_dir / f"{stem}_part{shard_index}.npy",
                tensor.numpy().astype(np.float16))
        buffered, n_buffered = [], 0
        shard_index += 1

    with get_reader(str(path)) as infile:
        reads = infile.iter_reads()
        if progress:
            try:
                from tqdm import tqdm
                reads = tqdm(reads, total=infile.get_total_reads(),
                             desc=f"  {Path(path).name}", leave=False)
            except ImportError:
                pass

        for read_id, signal_pa in reads:
            try:
                if len(signal_pa) < min_length:                 # too short to chunk
                    continue
                shift, scale = med_mad_normalisation(signal_pa)  # normalise whole read
                signal_norm = (signal_pa - shift) / scale
                clean = signal_norm[trim(signal_norm, threshold=threshold):]
                blocks = split_to_chunk(clean, chunk_size, overlap)
                if blocks is not None:
                    buffered.append(blocks)
                    n_buffered += blocks.shape[0]
                    n_chunks += blocks.shape[0]
                if n_buffered >= max_chunks_per_file:
                    flush()
            except Exception as exc:                            # one bad read must not stop the file
                n_failed += 1
                print(f"    ! read {read_id}: {exc}")

    flush()
    return shard_index, n_chunks, n_failed


def build_pretrain_chunks(raw_dir, out_dir=None, *, modality="dna_r9", min_length=16384,
                          chunk_size=16384, overlap=2048,
                          max_chunks_per_file=MAX_CHUNKS_PER_FILE,
                          recursive=False, resume=False, progress=True, workers=1):
    """Chunk every raw-signal file under ``raw_dir``.

    ``out_dir`` defaults to a ``pretrain_chunk`` directory next to ``raw_dir``.
    With ``resume`` set, files that already have a ``<stem>_part0.npy`` are skipped.
    ``workers`` > 1 processes whole files in parallel — they share nothing and each
    writes its own shards, so this is safe; per-file progress bars are turned off
    because interleaved bars are unreadable.

    Returns a summary dict.
    """
    raw_dir = Path(raw_dir)
    if out_dir is None:
        out_dir = raw_dir.parent / "pretrain_chunk"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_signal_files(raw_dir, recursive=recursive)
    print(f" * {len(files)} raw file(s) under {raw_dir}"
          f"{' (recursive)' if recursive else ''}")
    print(f" * writing {chunk_size}-sample chunks (overlap {overlap}) to {out_dir}")

    totals = {"files": 0, "skipped": 0, "shards": 0, "chunks": 0, "failed_reads": 0,
              "empty_files": []}

    todo = []
    for path in files:
        if resume and (out_dir / f"{path.stem}_part0.npy").exists():
            totals["skipped"] += 1
        else:
            todo.append(path)
    if totals["skipped"]:
        print(f" * {totals['skipped']} file(s) already done, skipping")

    opts = dict(modality=modality, min_length=min_length, chunk_size=chunk_size,
                overlap=overlap, max_chunks_per_file=max_chunks_per_file)

    def record(path, result):
        shards, chunks, failed = result
        if shards == 0:
            print(f" * {path.name} -> no chunks "
                  f"(every read shorter than {chunk_size} samples)")
            totals["empty_files"].append(path.name)
        else:
            print(f" * {path.name} -> {chunks} chunks in {shards} shard(s)")
        totals["files"] += 1
        totals["shards"] += shards
        totals["chunks"] += chunks
        totals["failed_reads"] += failed

    workers = max(1, int(workers))
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        workers = min(workers, len(todo))
        print(f" * {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(chunk_one_file, path, out_dir, progress=False, **opts): path
                       for path in todo}
            for future in as_completed(futures):
                record(futures[future], future.result())
    else:
        for path in todo:
            record(path, chunk_one_file(path, out_dir, progress=progress, **opts))

    print(f"\n * done: {totals['chunks']} chunks in {totals['shards']} shard(s) "
          f"from {totals['files']} file(s)"
          + (f", {totals['skipped']} skipped" if totals["skipped"] else "")
          + (f", {totals['failed_reads']} read(s) failed" if totals["failed_reads"] else ""))
    return totals

# ----------------------------------------------------------------------------- inference chunking
# Basecalling needs a DIFFERENT cut from pretraining, and the two must not be confused:
#   pretraining  drops the leading remainder, drops reads shorter than one chunk, and trims the
#                open-pore / stall region -- fine when all you want is clean signal to learn from.
#   inference    must cover every sample exactly once after stitching and must not lose a read,
#                so it pads short reads, prepends a chunk at offset 0 when the read does not
#                divide evenly, and does NOT trim (trimmed samples would be bases missing from
#                the output sequence).
# `stub` is returned because the stitcher needs it to place the first chunk's contribution.

def short_pad(sig, chunksize, mode="tile"):
    """Fill a read shorter than one chunk up to ``chunksize``.

    Only the first ``len(sig)/stride`` frames survive stitching, but the backend is
    bidirectional, so whatever fills the tail still leaks into the kept region. ``tile``
    is bonito's and dorado's default.
    """
    T = len(sig)
    pad = chunksize - T
    if mode == "tile":                       # bonito default: repeat the read
        n, overhang = divmod(chunksize, T)
        return np.concatenate([np.tile(sig, n), sig[:overhang]])
    if mode == "zero":                       # 0 == the normalised signal's median level
        return np.concatenate([sig, np.zeros(pad, sig.dtype)])
    if mode == "reflect":                    # mirror: no seam discontinuity at the read end
        if pad < T:
            return np.pad(sig, (0, pad), mode="reflect")
        return np.pad(np.tile(sig, -(-chunksize // T))[:chunksize], (0, 0))
    if mode == "noise":                      # open-pore-like: gaussian with the read's own stats
        rng = np.random.default_rng(0)
        return np.concatenate([sig, rng.normal(np.median(sig), np.std(sig), pad).astype(sig.dtype)])
    raise ValueError(f"unknown pad mode: {mode}")


def chunk_for_inference(sig, chunksize=16384, overlap=2048, pad_mode="tile"):
    """Cut one normalised read into overlapping chunks that cover all of it.

    Returns ``(chunks [n, chunksize] float32, stub)``. ``stub`` is the number of leading
    samples that do not fit the regular stride; when it is non-zero an extra chunk starting
    at sample 0 is prepended so the read start is covered. Reads shorter than one chunk are
    padded by :func:`short_pad` and returned as a single chunk.
    """
    T = len(sig)
    if T < chunksize:
        return short_pad(sig, chunksize, pad_mode).astype(np.float32)[None, :], 0
    stub = (T - overlap) % (chunksize - overlap)
    starts = np.arange(stub, T - chunksize + 1, chunksize - overlap)
    chunks = np.stack([sig[s:s + chunksize] for s in starts])
    if stub > 0:
        chunks = np.concatenate([sig[None, :chunksize], chunks], 0)
    return chunks.astype(np.float32), stub
