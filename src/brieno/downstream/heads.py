"""Classification heads for the per-chunk downstream tasks.

Every head is the same :class:`ChunkHead`; they differ only in what is fed to it. That
is deliberate — it is what makes "does pretraining help?" answerable, because the
comparison is between inputs rather than between architectures:

===== ============================================================================
 head  input
===== ============================================================================
 A     the tokenizer's continuous encoder latent
 B1    the quantized latent (what the codebooks reconstruct)
 B2    the token ids alone, through a learned embedding
 C1    token ids through the pretrained transformer, frozen
 C2    the same, with the transformer fine-tuned along with the head
 D1    the tokenizer's encoder architecture, randomly initialised and trained from
       scratch on the task — no tokenizer, no transformer, no pretraining
===== ============================================================================

The number of classes comes from the dataset, so the same code serves barcode
demultiplexing (24 classes) and species identification (10) without changing.
"""
import math

import torch
import torch.nn as nn

__all__ = ["AttnPool", "ChunkHead", "CodebookEmbedding", "RawEncoder", "Heads"]


class AttnPool(nn.Module):
    """Attention pooling with one learned query: [B, L, d] -> [B, d]."""

    def __init__(self, d, n_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(self.q.expand(x.shape[0], -1, -1), x, x)
        return out[:, 0]


class ChunkHead(nn.Module):
    """[B, L, in_dim] -> class logits.

    Linear projection, three stride-2 convolutions, then attention pooling concatenated
    with a mean — the two pooled views answer different questions ("where is the most
    informative frame" and "what does the chunk look like overall"), and concatenating
    them costs almost nothing. About 1.1 M parameters at ``d=256``.
    """

    def __init__(self, in_dim, n_classes, d=256, n_conv=3, kernel=5, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        layers = []
        for _ in range(n_conv):
            layers += [nn.Conv1d(d, d, kernel, stride=2, padding=kernel // 2),
                       nn.GroupNorm(8, d), nn.GELU(), nn.Dropout(dropout)]
        self.conv = nn.Sequential(*layers)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x, return_emb=False):
        x = self.proj(x).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        e = torch.cat([self.pool(x), x.mean(1)], -1)          # [B, 2d] chunk embedding
        return (self.head(e), e) if return_emb else self.head(e)


class CodebookEmbedding(nn.Module):
    """Token ids [B, Q, L] -> [B, L, d], for the head that sees ids and nothing else.

    ``concat=True`` gives each codebook its own ``d // Q`` slice instead of summing
    them. Summing lets one codebook's embedding cancel another's; keeping them in
    separate slices does not, which matters when the codebooks carry different
    information rather than refinements of the same thing.
    """

    def __init__(self, n_q, n_tokens, d, concat=False):
        super().__init__()
        self.concat = bool(concat)
        dq = d // n_q if self.concat else d
        assert not self.concat or dq * n_q == d, f"d={d} must divide by n_q={n_q}"
        self.emb = nn.ModuleList([nn.Embedding(n_tokens, dq) for _ in range(n_q)])
        self.w = nn.Parameter(torch.ones(n_q))

    def forward(self, idx):
        if self.concat:
            return torch.cat([self.emb[q](idx[:, q]) for q in range(len(self.emb))], -1)
        return sum(self.w[q] * self.emb[q](idx[:, q]) for q in range(len(self.emb)))


class RawEncoder(nn.Module):
    """The tokenizer's encoder architecture, randomly initialised and trained on the task.

    This is the control for the whole premise: same encoder shape, same head, same data,
    but no tokenizer, no transformer and no pretraining. Whatever the pretrained heads
    beat it by is what pretraining bought.
    """

    def __init__(self, hidden_dim, downsample, dropout=0.1):
        super().__init__()
        from brieno.models.tokenizer import SignalEncoder
        self.enc = SignalEncoder(1, hidden_dim, dropout, downsample=downsample)

    def forward(self, x):
        return self.enc(x)                                     # [B, D, L]


class Heads(nn.Module):
    """Any subset of A / B1 / B2 / C1 / C2 / D1, each the same :class:`ChunkHead`.

    Several heads can share one forward pass over the frozen front-end, which is the
    cheap way to compare them: the expensive part is computed once per batch.
    """

    CONTEXT_HEADS = ("C1", "C2")

    def __init__(self, names, hidden_dim, n_q, n_tokens, n_classes, d=256,
                 b2_concat=False, ctx_dim=None):
        """``hidden_dim`` is the tokenizer's width (A / B1 / D1); ``ctx_dim`` the
        transformer's (C1 / C2). They are equal in the released models, but assuming so
        would silently build the wrong head the first time they are not.
        """
        super().__init__()
        self.names = list(names)
        ctx_dim = hidden_dim if ctx_dim is None else int(ctx_dim)
        self.h = nn.ModuleDict()
        for n in self.names:
            if n == "B2":
                self.B2_emb = CodebookEmbedding(n_q, n_tokens, d, concat=b2_concat)
                self.h[n] = ChunkHead(d, n_classes, d)
            elif n in self.CONTEXT_HEADS:
                self.h[n] = ChunkHead(ctx_dim, n_classes, d)
            else:
                self.h[n] = ChunkHead(hidden_dim, n_classes, d)

    def forward(self, feats, return_emb=False):
        out, emb = {}, {}
        for n in self.names:
            if n in ("A", "B1", "D1"):
                r = self.h[n](feats[n].transpose(1, 2), return_emb=return_emb)
            elif n == "B2":
                r = self.h[n](self.B2_emb(feats["B2"]), return_emb=return_emb)
            else:                                              # C1 / C2: already [B, L, D]
                r = self.h[n](feats[n], return_emb=return_emb)
            if return_emb:
                out[n], emb[n] = r
            else:
                out[n] = r
        return (out, emb) if return_emb else out

    def n_params(self, n):
        total = sum(p.numel() for p in self.h[n].parameters())
        if n == "B2":
            total += sum(p.numel() for p in self.B2_emb.parameters())
        return total

    def head_state(self, n):
        state = {"head": self.h[n].state_dict()}
        if n == "B2":
            state["B2_emb"] = self.B2_emb.state_dict()
        return state

    def load_head_state(self, n, state):
        self.h[n].load_state_dict(state["head"])
        if n == "B2" and "B2_emb" in state:
            self.B2_emb.load_state_dict(state["B2_emb"])
