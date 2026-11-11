"""The frozen front-end the task heads attach to.

A chunk of raw signal passes through the tokenizer once and yields three views — the
continuous latent, the quantized latent and the token ids — and optionally through the
transformer for a contextual hidden state. Everything a head might want is produced in
one pass, so comparing heads costs one forward, not one per head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["load_transformer", "transformer_hidden", "context_input", "FrozenFeatures",
           "FrozenStack", "TransformerHidden"]


def _bf16():
    """bf16 autocast where the GPU supports it; a no-op elsewhere (V100, CPU)."""
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def load_transformer(checkpoint_path, device, trainable=False, random_init=False):
    """Rebuild the token transformer from a checkpoint that carries its own architecture.

    The architecture includes the input mode, so a transformer that reads code ids and
    one that reads the continuous latent load through the same call and the caller
    need not know in advance which it has been given.

    ``random_init`` keeps the architecture but not the weights: the "no pretraining"
    control for a fine-tuned transformer, which is the only honest comparison for
    claims about what pretraining contributes.
    """
    from brieno.models.transformer import TokenTransformerMLM

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in state:
        raise ValueError(f"{checkpoint_path} carries no 'model_config'; it is not a "
                         "brieno transformer checkpoint")
    m = state["model_config"]
    tf = TokenTransformerMLM.from_config(m)
    if not random_init:
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in state["model_state_dict"].items()}
        tf.load_state_dict(sd, strict=True)
    for p in tf.parameters():
        p.requires_grad_(trainable)
    tf = tf.to(device)
    tf.train(trainable)
    return tf, m


def transformer_hidden(tf, x, cb_weights=None):
    """The transformer's input -> last-layer residual stream [B, L, D].

    ``x`` is whatever the transformer reads: code ids [B, Q, L] for a token-mode model,
    the continuous latent [B, L, latent_dim] for a latent-mode one. Nothing is masked —
    this is the representation of the whole chunk, as a head sees it.

    The model is pre-LN and has no final norm of its own (the MLM heads supply it), so
    a parameter-free LayerNorm is applied here; without it the scale drifts with depth
    and the head has to absorb it.

    Gradients flow into ``tf`` when its parameters are trainable, which is the only
    difference between the frozen and fine-tuned heads.
    """
    h = tf.hidden(x, cb_weights=cb_weights)
    return F.layer_norm(h.float(), (h.shape[-1],))


def context_input(tf, latent, tokens):
    """Pick what ``tf`` reads out of the tokenizer's two views of a chunk.

    ``latent`` is channel-first [B, D, L] as the tokenizer's encoder emits it; a
    latent-mode transformer wants it sequence-first.
    """
    return tokens if tf.input_mode == "token" else latent.transpose(1, 2)


class FrozenFeatures(nn.Module):
    """Raw chunk [B, 1, 16384] -> (latent, quantized latent, token ids), all detached.

    The tokenizer stays in eval mode and under ``no_grad``: the heads are what is being
    trained, and a tokenizer that drifted during fine-tuning would make the comparison
    between heads meaningless.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tok = tokenizer

    @torch.no_grad()
    def forward(self, x):
        with _bf16():
            lat = self.tok.encoder(x)
        lat = lat.float()
        with torch.autocast("cuda", enabled=False):
            # quantize(), not quantizer(): a low-dimensional quantizer's projections
            # belong to the former, and skipping them silently changes the features.
            q, idx, _, _ = self.tok.quantize(lat)
        return lat, q, idx.permute(1, 0, 2).long()


class FrozenStack(nn.Module):
    """The whole frozen front-end as one module, so `nn.DataParallel` can split a batch.

    Everything inside runs under ``no_grad``, so parallelism here only scatters and
    gathers activations; no gradients are synchronised. The trainable head stays on
    one device.
    """

    def __init__(self, feat, tf=None, cb_w=None):
        super().__init__()
        self.feat = feat
        self.tf = tf
        self.cb_w = cb_w

    @torch.no_grad()
    def forward(self, x):
        lat, q, idx = self.feat(x)
        if self.tf is None:
            return lat, q, idx
        with _bf16():
            h = transformer_hidden(self.tf, context_input(self.tf, lat, idx), self.cb_w)
        return lat, q, idx, h.float()


class TransformerHidden(nn.Module):
    """The fine-tuned transformer as a module, so its gradients can be reduced across GPUs.

    Only the fine-tuned head needs this; the frozen one rides inside
    :class:`FrozenStack` where nothing requires grad.
    """

    def __init__(self, tf, cb_w=None):
        super().__init__()
        self.tf = tf
        self.cb_w = cb_w

    def forward(self, x):
        with _bf16():
            return transformer_hidden(self.tf, x, self.cb_w).float()
