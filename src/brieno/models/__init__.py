"""Model definitions: signal tokenizer, token transformer, CRF basecalling head."""
from brieno.models.tokenizer import (
    SignalTokenizer,
    SignalEncoder,
    SignalDecoder,
    ResidualVectorQuantizer,
    EMAVectorQuantizer,
    GroupedFSQ,
)
from brieno.models.transformer import TokenTransformerMLM
from brieno.models.vocab import (
    BASE_TO_ID,
    ID_TO_BASE,
    BLANK_ID,
    BASE_VOCAB_SIZE,
    encode_ref,
    decode_ids,
)

__all__ = [
    "SignalTokenizer",
    "SignalEncoder",
    "SignalDecoder",
    "ResidualVectorQuantizer",
    "EMAVectorQuantizer",
    "GroupedFSQ",
    "TokenTransformerMLM",
    "BASE_TO_ID",
    "ID_TO_BASE",
    "BLANK_ID",
    "BASE_VOCAB_SIZE",
    "encode_ref",
    "decode_ids",
]

# brieno.models.crf is NOT imported here: it probes for a CUDA backend (koi / cupy)
# at import time. Import it explicitly where the CRF head is actually used.
