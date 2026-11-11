"""Chunk-level basecalling accuracy: a decoded chunk aligned against its reference.

The headline number is the identity of a global (Needleman-Wunsch) alignment scored
match 2 / mismatch -1 / gap open 8 / gap extend 4, the training-time metric of
Pages-Gallego & de Ridder (2023). A local (Smith-Waterman) identity is reported next
to it: it forgives a few bases of overshoot at the chunk boundary, which the global
alignment counts as insertions.

``identity = matches / (matches + mismatches + insertions + deletions)``.

Alignment uses parasail when it is installed and falls back to edlib with unit costs
otherwise. The two do not give the same numbers, so :data:`METRIC` records which one
produced a result and belongs next to every number that is reported.
"""
import re

import numpy as np

try:
    import parasail
    _HAVE_PARASAIL = True
except ImportError:
    _HAVE_PARASAIL = False

__all__ = ["METRIC", "chunk_identity", "ChunkScores"]

METRIC = "parasail_nw_2_-1_8_4" if _HAVE_PARASAIL else "edlib_nw_unit_costs"

_CIGAR = re.compile(r"(\d+)([=XIDSMN])")
_NW_MATRIX = None


def _nw_matrix():
    global _NW_MATRIX
    if _NW_MATRIX is None:                       # allocates a C struct: build it once
        _NW_MATRIX = parasail.matrix_create("ACGT", 2, -1)
    return _NW_MATRIX


def _counts(cigar):
    n = {"=": 0, "X": 0, "I": 0, "D": 0}
    for count, op in _CIGAR.findall(cigar):
        if op == "M":                            # edlib without extended ops
            op = "="
        if op in n:
            n[op] += int(count)
    return n


def chunk_identity(pred, ref, mode="nw"):
    """Align ``pred`` to ``ref``; ``mode`` is ``"nw"`` (global) or ``"sw"`` (local).

    Returns ``None`` when either string is empty or nothing aligns, otherwise a dict
    with ``identity``, ``len_ratio`` and the operation counts ``n_match``, ``n_mis``,
    ``n_ins``, ``n_del``.
    """
    if mode not in ("nw", "sw"):
        raise ValueError(f"mode must be 'nw' or 'sw', got {mode!r}")
    if not pred or not ref:
        return None

    if _HAVE_PARASAIL:
        if mode == "nw":
            aln = parasail.nw_trace_striped_32(pred, ref, 8, 4, _nw_matrix())
        else:
            aln = parasail.sw_trace_striped_32(pred, ref, 8, 4, parasail.dnafull)
        n = _counts(aln.cigar.decode.decode())
    else:
        import edlib
        res = edlib.align(pred, ref, mode="NW" if mode == "nw" else "HW", task="path")
        if res["cigar"] is None:
            return None
        n = _counts(res["cigar"])

    aligned = n["="] + n["X"] + n["I"] + n["D"]
    if aligned == 0:
        return None
    return {"identity": n["="] / aligned, "len_ratio": len(pred) / max(1, len(ref)),
            "n_match": n["="], "n_mis": n["X"], "n_ins": n["I"], "n_del": n["D"]}


class ChunkScores:
    """Running chunk-level scores for one head over one evaluation set.

    A chunk that cannot be aligned at all (an empty call, usually) counts as identity
    zero rather than being dropped: leaving failures out would reward a model for
    producing nothing on the chunks it finds hard.
    """

    def __init__(self):
        self.nw, self.sw, self.len_ratio = [], [], []
        self.n_match = self.n_mis = self.n_ins = self.n_del = self.n_fail = 0

    def add(self, preds, refs):
        for pred, ref in zip(preds, refs):
            nw = chunk_identity(pred, ref, "nw")
            sw = chunk_identity(pred, ref, "sw")
            if nw is None or sw is None:
                self.nw.append(0.0)
                self.sw.append(0.0)
                self.len_ratio.append(len(pred) / max(1, len(ref)))
                self.n_fail += 1
                continue
            self.nw.append(nw["identity"])
            self.sw.append(sw["identity"])
            self.len_ratio.append(nw["len_ratio"])
            self.n_match += nw["n_match"]
            self.n_mis += nw["n_mis"]
            self.n_ins += nw["n_ins"]
            self.n_del += nw["n_del"]

    def summary(self):
        nw, sw = np.array(self.nw), np.array(self.sw)
        aligned = max(1, self.n_match + self.n_mis + self.n_ins + self.n_del)
        return {
            "n": int(len(nw)),
            "nw_mean": float(nw.mean()), "nw_median": float(np.median(nw)),
            "nw_p10": float(np.percentile(nw, 10)),
            "sw_mean": float(sw.mean()), "sw_median": float(np.median(sw)),
            "len_ratio": float(np.mean(self.len_ratio)),
            "identity_agg": self.n_match / aligned,
            "mismatch_rate": self.n_mis / aligned,
            "ins_rate": self.n_ins / aligned,
            "del_rate": self.n_del / aligned,
            "n_fail": int(self.n_fail),
        }
