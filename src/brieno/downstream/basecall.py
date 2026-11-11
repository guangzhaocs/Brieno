"""The basecalling head: a fixed sequence back-end with a CTC-CRF output layer.

Every basecalling head is the same back-end on a different representation, exactly as
the classification heads share one :class:`~brieno.downstream.heads.ChunkHead`. What a
head is fed is the only thing that changes between ``A``, ``B1``, ``B2``, ``C1``, ``C2``
and ``D1``; see :mod:`brieno.downstream.heads` for what each letter means.

The output layer, loss and decoder are those of Dorado and Bonito
(:mod:`brieno.models.crf`), so a difference in accuracy against them comes from what
is underneath the output layer and not from how sequences are scored.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from brieno.downstream.heads import CodebookEmbedding

__all__ = ["SAMPLES_PER_FRAME", "backend_upsample", "SeqBackend", "BasecallHeads",
           "pad_targets"]

#: Every head emits one frame of CRF scores per this many raw samples, whatever the
#: tokenizer's own stride. Dorado's R9.4.1 models use 5.
SAMPLES_PER_FRAME = 4

#: Width of the learned embedding the token-id head (B2) reads through.
B2_EMBED_DIM = 256


def backend_upsample(downsample):
    """How much the back-end must upsample so the output has one frame per
    :data:`SAMPLES_PER_FRAME` samples: a 16x tokenizer needs 4x, an 8x one 2x, a 4x one none.
    """
    downsample = int(downsample)
    if downsample not in (4, 8, 16):
        raise ValueError(f"tokenizer downsample must be 4, 8 or 16, got {downsample}")
    return downsample // SAMPLES_PER_FRAME


class SeqBackend(nn.Module):
    """``[B, L, in_dim]`` features -> CRF scores ``[T, B, 5 * 4**state_len]``, ``T = upsample * L``.

    Linear projection to ``d``, transposed convolutions up to the output frame rate, a
    bidirectional LSTM, then the CRF output layer. The defaults are the configuration
    the released basecalling models were trained with (``d`` 512, 5 x 512 LSTM, no
    dropout).

    ``n_lstm = 0`` removes the sequence model, leaving upsampling and the output layer:
    whatever context the head then has must come from the representation it is fed.
    """

    def __init__(self, in_dim, d=512, lstm_hidden=512, n_lstm=5, dropout=0.0, state_len=5,
                 crf_scale=5.0, blank_score=2.0, upsample=2):
        super().__init__()
        from brieno.models.crf import LinearCRF      # probes for a CUDA backend on import

        if upsample not in (1, 2, 4):
            raise ValueError(f"upsample must be 1, 2 or 4, got {upsample}")
        self.state_len, self.upsample = state_len, upsample
        self.proj = nn.Linear(in_dim, d)
        # 4x: two stride-2 transposed convolutions; 2x: the first only; 1x: none
        self.up1 = nn.ConvTranspose1d(d, d, 4, stride=2, padding=1) if upsample >= 2 else None
        self.n1 = nn.GroupNorm(8, d) if upsample >= 2 else None
        self.up2 = nn.ConvTranspose1d(d, d, 4, stride=2, padding=1) if upsample == 4 else None
        self.n2 = nn.GroupNorm(8, d) if upsample == 4 else None
        self.lstm = nn.LSTM(d, lstm_hidden, num_layers=n_lstm, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_lstm > 1 else 0.0) if n_lstm > 0 else None
        self.drop = nn.Dropout(dropout)
        out_in = 2 * lstm_hidden if n_lstm > 0 else d
        self.head = LinearCRF(out_in, state_len=state_len, scale=crf_scale,
                              blank_score=blank_score)

    def forward(self, x):                                       # [B, L, in_dim]
        h = self.proj(x).transpose(1, 2)                        # [B, d, L]
        if self.up1 is not None:
            h = F.gelu(self.n1(self.up1(h)))
        if self.up2 is not None:
            h = F.gelu(self.n2(self.up2(h)))                    # [B, d, upsample * L]
        h = h.transpose(1, 2).contiguous()                      # [B, T, d]
        if self.lstm is not None:
            h, _ = self.lstm(h)                                 # [B, T, 2 * lstm_hidden]
        h = self.drop(h)
        return self.head(h.float())                             # [T, B, C], fp32 for the loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class BasecallHeads(nn.Module):
    """Any subset of A / B1 / B2 / C1 / C2 / D1, each the same :class:`SeqBackend`.

    ``hidden_dim`` is the tokenizer's width (A / B1 / D1) and ``ctx_dim`` the
    transformer's (C1 / C2); B2 owns the embedding its token ids are read through.
    """

    CONTEXT_HEADS = ("C1", "C2")

    def __init__(self, names, hidden_dim, n_q, n_tokens, ctx_dim=None, **backend_kw):
        super().__init__()
        self.names = list(names)
        ctx_dim = hidden_dim if ctx_dim is None else int(ctx_dim)
        self.h = nn.ModuleDict()
        for n in self.names:
            if n == "B2":
                self.B2_emb = CodebookEmbedding(n_q, n_tokens, B2_EMBED_DIM)
                self.h[n] = SeqBackend(B2_EMBED_DIM, **backend_kw)
            elif n in self.CONTEXT_HEADS:
                self.h[n] = SeqBackend(ctx_dim, **backend_kw)
            else:
                self.h[n] = SeqBackend(hidden_dim, **backend_kw)

    def forward(self, feats):
        out = {}
        for n in self.names:
            if n in ("A", "B1", "D1"):
                out[n] = self.h[n](feats[n].transpose(1, 2).float())    # [B, D, L] -> [B, L, D]
            elif n == "B2":
                out[n] = self.h[n](self.B2_emb(feats["B2"]))
            else:                                                       # C1 / C2: already [B, L, D]
                out[n] = self.h[n](feats[n].float())
        return out

    def parameters_of(self, n):
        """The parameters that belong to head ``n`` alone (its back-end, and B2's embedding)."""
        params = list(self.h[n].parameters())
        if n == "B2":
            params += list(self.B2_emb.parameters())
        return params

    def n_params(self, n):
        return sum(p.numel() for p in self.parameters_of(n))

    def head_state(self, n):
        state = {"head": self.h[n].state_dict()}
        if n == "B2":
            state["B2_emb"] = self.B2_emb.state_dict()
        return state

    def load_head_state(self, n, state):
        self.h[n].load_state_dict(state["head"])
        if n == "B2":
            self.B2_emb.load_state_dict(state["B2_emb"])


def pad_targets(targets_flat, target_lengths):
    """Concatenated targets -> padded ``[N, U]`` (0 = pad), the layout the CRF loss wants."""
    parts = torch.split(targets_flat, target_lengths.tolist())
    return nn.utils.rnn.pad_sequence(parts, batch_first=True, padding_value=0)
