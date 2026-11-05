"""Basecalling alphabet shared by every CRF/CTC head in Brieno.

4 bases, no ``N``: a basecaller that can emit ``N`` learns "when the signal looks
like X, output N", which actively harms its 4-base discrimination. Bonito, Dorado
and RODAN all use a 4-base vocabulary, and downstream tools (minimap2, variant
callers) expect ACGT output. Chunks whose reference contains any ``N`` are dropped
at dataset-load time instead.
"""
import torch

BASE_TO_ID = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
ID_TO_BASE = {v: k for k, v in BASE_TO_ID.items()}
BLANK_ID = 0
BASE_VOCAB_SIZE = len(BASE_TO_ID)          # 4 non-blank classes


def encode_ref(ref_str: str) -> torch.LongTensor:
    """ACGT string -> 1-D LongTensor of ids (0 is the CTC blank, never produced here).

    Raises KeyError on any non-ACGT character, ``N`` included; callers must drop
    N-containing references first.
    """
    return torch.tensor([BASE_TO_ID[c] for c in ref_str], dtype=torch.long)


def decode_ids(ids) -> str:
    """Inverse of :func:`encode_ref`, stripping blanks."""
    return ''.join(ID_TO_BASE[int(i)] for i in ids if int(i) != BLANK_ID)
