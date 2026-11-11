import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TokenTransformerMLM(nn.Module):
    """
    Masked-token prediction over tokenized signal.

    The prediction target is always the same: the RVQ code ids of the hidden frames,
    one head per codebook. What differs is what the model reads, set by ``input_mode``:

    ``"token"`` (default)
        input_ids : [Batch, num_vq_layers, Seq_len] code ids, summed over codebooks
        through one embedding table each. Hidden frames carry MASK_TOKEN_ID. The
        quantizer sits in the input path: the model only ever sees what the codebooks
        kept.

    ``"latent"``
        latent : [Batch, Seq_len, latent_dim], the tokenizer's encoder output before
        quantization, through LayerNorm and a linear projection. Hidden frames are
        replaced by a learned mask vector. The quantizer sits in the label path only,
        and is not needed at inference.

    Output (both modes):
        logits_list : List of num_vq_layers tensors, each [Batch, Seq_len, vocab_size]
                      Head q predicts the code id of codebook q.

    Design notes:
        - All codebooks are masked at the SAME positions.
        - Head q classifies against its own table ``token_embeddings[q]``. In token mode
          that table is also the input embedding (weight tying); in latent mode there is
          no token input, so it serves as the output prototypes only.
        - Loss is computed only at masked positions, averaged across codebooks.
        - MASK_TOKEN_ID = vocab_size,     PAD_TOKEN_ID = vocab_size + 1
        - A token-mode model has exactly the parameters it had before ``input_mode``
          existed, so earlier checkpoints load unchanged.
    """

    INPUT_MODES = ("token", "latent")

    # Special token IDs (fixed, not learnable)
    MASK_TOKEN_ID: int
    PAD_TOKEN_ID:  int

    def __init__(
        self,
        vocab_size:   int = 1024,
        d_model:      int = 768,
        nhead:        int = 12,
        num_layers:   int = 6,
        max_seq_len:  int = 2048,
        num_vq_layers: int = 4,
        dropout:      float = 0.1,
        input_mode:   str = "token",
        latent_dim:   Optional[int] = None,
    ):
        super().__init__()

        if input_mode not in self.INPUT_MODES:
            raise ValueError(f"input_mode must be one of {self.INPUT_MODES}, got {input_mode!r}")
        if input_mode == "latent" and latent_dim is None:
            raise ValueError("input_mode='latent' needs latent_dim, the width of the "
                             "tokenizer's encoder output")

        self.vocab_size    = vocab_size
        self.num_vq_layers = num_vq_layers
        self.d_model       = d_model
        self.input_mode    = input_mode
        self.latent_dim    = int(latent_dim) if input_mode == "latent" else None

        # Special token IDs
        self.MASK_TOKEN_ID = vocab_size        # 1024
        self.PAD_TOKEN_ID  = vocab_size + 1    # 1025
        actual_vocab_size  = vocab_size + 2    # embeddings cover real + MASK + PAD

        # ── 1. Token Embeddings (one per VQ layer) ──────────────────────────
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(actual_vocab_size, d_model)
            for _ in range(num_vq_layers)
        ])
        for emb in self.token_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # ── 2. Learned Positional Encoding ───────────────────────────────────
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # ── 3. Input projection: LayerNorm + Dropout ─────────────────────────
        self.input_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.input_dropout    = nn.Dropout(dropout)

        # ── 4. Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            norm_first=True,          # Pre-LN: more stable training
            batch_first=True,         # [Batch, Seq, Dim]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── 5. LM Heads: one per VQ layer, with weight tying ─────────────────
        # Weight tying: each head shares the embedding matrix of its own layer.
        # Only bias is a learnable parameter; the weight is borrowed from the
        # corresponding token_embeddings (only the first vocab_size rows,
        # excluding MASK and PAD rows).
        self.lm_head_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(vocab_size))
            for _ in range(num_vq_layers)
        ])

        # ── 6. Continuous input (latent mode only) ───────────────────────────
        # Created last, and only in latent mode: a token-mode model must keep both its
        # state_dict keys and the order in which it consumes random numbers at
        # initialisation, or published weights and seeded runs stop reproducing.
        if input_mode == "latent":
            self.input_norm = nn.LayerNorm(self.latent_dim)
            self.input_proj = nn.Linear(self.latent_dim, d_model)
            self.mask_embedding = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

    # ── construction from a config ───────────────────────────────────────────
    @classmethod
    def from_config(cls, m: dict, **overrides):
        """Build from the ``model`` section of a training config or a checkpoint's
        ``model_config``. A config without ``input_mode`` describes a token-mode model,
        which is what every config written before the option existed means."""
        kw = dict(
            vocab_size=int(m["vocab_size"]),
            d_model=int(m["d_model"]),
            nhead=int(m["n_head"]),
            num_layers=int(m["n_layer"]),
            max_seq_len=int(m["max_seq_len"]),
            num_vq_layers=int(m.get("num_vq_layers", 4)),
            dropout=float(m.get("dropout", 0.1)),
            input_mode=m.get("input_mode", "token"),
            latent_dim=m.get("latent_dim"),
        )
        kw.update(overrides)
        return cls(**kw)

    # ── codebook initialisation (2026-09-22) ─────────────────────────────────
    @torch.no_grad()
    def init_embeddings_from_codebooks(self, codebooks, target_std: float = 0.02, freeze: bool = False):
        """Initialise token_embeddings[q][:vocab_size] from the tokenizer's RVQ codebooks E_q (list of [vocab_size, d_model]).
        All layers are scaled by ONE common factor so that the layer with the largest spread (codebook 0) has std=target_std:
        the geometry of the summed embedding sum_q E_q[id_q] (= the tokenizer's quantized latent) is preserved exactly, and the
        weight-tied prediction heads start from the same (rescaled) codebook space. MASK / PAD rows keep their random init.
        freeze=True keeps the codebook rows fixed (the tied LM heads then only train their bias)."""
        assert len(codebooks) == self.num_vq_layers, (len(codebooks), self.num_vq_layers)
        ref_std = float(codebooks[0].float().std()); scale = target_std / ref_std
        stats = []
        for q, cb in enumerate(codebooks):
            cb = cb.detach().float().to(self.token_embeddings[q].weight.device)
            assert cb.shape == (self.vocab_size, self.d_model), (q, tuple(cb.shape), self.vocab_size, self.d_model)
            self.token_embeddings[q].weight[:self.vocab_size].copy_(cb * scale)
            stats.append((q, float(cb.std()), float((cb * scale).std()), float((cb * scale).norm(dim=-1).mean())))
        if freeze:
            for emb in self.token_embeddings: emb.weight.requires_grad_(False)
        return dict(scale=scale, ref_std=ref_std, freeze=freeze, per_layer=stats)

    # ── input embedding ──────────────────────────────────────────────────────
    def embed(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
              cb_weights: Optional[List[float]] = None) -> torch.Tensor:
        """The model's input, either mode -> [Batch, Seq_len, d_model] entering the encoder.

        Token mode
            x          : [B, Q, L] code ids (MASK_TOKEN_ID at hidden frames).
            cb_weights : optional fixed per-codebook input weights; a zero drops that
                         codebook from the input. Default: all codebooks, unweighted.
        Latent mode
            x    : [B, L, latent_dim] continuous latent.
            mask : optional [B, L] bool; True frames are replaced by the learned mask
                   vector. Leave None at inference.
        """
        if self.input_mode == "token":
            B, Q, L = x.shape
            assert Q == self.num_vq_layers, (
                f"input_ids has {Q} VQ layers but model expects {self.num_vq_layers}"
            )
            if cb_weights is None:
                # Each codebook contributes equally; summing is the standard approach
                # (same as combining multiple token types in BERT).
                e = self.token_embeddings[0](x[:, 0, :])                # [B, L, D]
                for q in range(1, self.num_vq_layers):
                    e = e + self.token_embeddings[q](x[:, q, :])
            else:
                assert len(cb_weights) == Q, (
                    f"cb_weights has {len(cb_weights)} entries, model has {Q} codebooks")
                e = None
                for q in range(Q):
                    if cb_weights[q] == 0:
                        continue
                    t = self.token_embeddings[q](x[:, q])
                    if cb_weights[q] != 1:
                        t = t * cb_weights[q]
                    e = t if e is None else e + t
        else:
            B, L, D = x.shape
            assert D == self.latent_dim, (
                f"latent has width {D} but model expects {self.latent_dim}"
            )
            e = self.input_proj(self.input_norm(x))                     # [B, L, D]
            if mask is not None:
                e = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(e.dtype), e)

        assert L <= self.pos_embedding.shape[1], (
            f"Sequence length {L} exceeds max_seq_len {self.pos_embedding.shape[1]}"
        )
        e = e + self.pos_embedding[:, :L, :]
        e = self.input_layer_norm(e)
        return self.input_dropout(e)

    def hidden(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
               src_key_padding_mask: Optional[torch.Tensor] = None,
               cb_weights: Optional[List[float]] = None) -> torch.Tensor:
        """Input -> last-layer residual stream [Batch, Seq_len, d_model].

        This is what a downstream head attaches to. The model is pre-LN, so the stream
        is not normalised here; the prediction heads and downstream code each apply
        their own.
        """
        return self.transformer(self.embed(x, mask=mask, cb_weights=cb_weights),
                                src_key_padding_mask=src_key_padding_mask)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x:           torch.Tensor,                    # token: [B, Q, L] ids; latent: [B, L, latent_dim]
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, L] bool, True = pad
        mask:        Optional[torch.Tensor] = None,   # latent mode: [B, L] bool, True = hidden frame
    ) -> List[torch.Tensor]:
        """
        Args:
            x                     : the model's input in its ``input_mode`` (see class docstring)
            src_key_padding_mask  : [Batch, Seq_len] bool (True = position to ignore)
                                    Pass this when sequences are padded to equal length.
            mask                  : latent mode only — frames to replace by the mask vector.
                                    Token mode marks hidden frames inside ``x`` itself.

        Returns:
            logits_list : list of num_vq_layers tensors, each [Batch, Seq_len, vocab_size]
                          Logits over vocab_size real tokens (MASK / PAD not predicted).
        """
        hidden_states = self.hidden(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # ── per-codebook prediction heads ─────────────────────────────────────
        # For codebook q: logits = hidden_states @ E_q[:vocab_size].T + bias_q
        # where E_q is the table of codebook q. MASK and PAD rows (indices vocab_size
        # and vocab_size+1) are excluded — the model predicts only real tokens.
        logits_list = []
        for q in range(self.num_vq_layers):
            tied_weight = self.token_embeddings[q].weight[:self.vocab_size, :]  # [V, D]
            logits_q = F.linear(hidden_states, tied_weight, self.lm_head_biases[q])
            logits_list.append(logits_q)    # [B, L, V]

        return logits_list

    # ── loss helper ───────────────────────────────────────────────────────────
    def compute_mlm_loss(
        self,
        logits_list:    List[torch.Tensor],   # num_vq_layers × [B, L, V]
        labels:         torch.Tensor,         # [B, Q, L]  -100 = ignore
        layer_weights:  Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compute mean cross-entropy loss across all VQ layers at masked positions.

        Args:
            logits_list   : output of forward()
            labels        : same shape as input_ids; -100 at non-masked positions
                            (standard HuggingFace convention — ignored by CE loss)
            layer_weights : optional per-layer loss weights, e.g. [1.0, 1.0, 0.5, 0.5]
                            to emphasise lower layers. Defaults to equal weights.

        Returns:
            scalar loss tensor
        """
        if layer_weights is None:
            layer_weights = [1.0] * self.num_vq_layers

        total_loss   = torch.tensor(0.0, device=logits_list[0].device)
        weight_sum   = sum(layer_weights)

        for q, (logits_q, w) in enumerate(zip(logits_list, layer_weights)):
            # logits_q : [B, L, V]  →  [B*L, V]
            # labels_q : [B, L]     →  [B*L]
            labels_q = labels[:, q, :]                          # [B, L]
            loss_q   = F.cross_entropy(
                logits_q.reshape(-1, self.vocab_size),
                labels_q.reshape(-1).long(),
                ignore_index=-100,
            )
            total_loss = total_loss + w * loss_q

        return total_loss / weight_sum


# ── unit test ─────────────────────────────────────────────────────────────────
def test_model_dimensions():
    print("=" * 60)
    print("TokenTransformerMLM — dimension & forward pass test")
    print("=" * 60)

    batch_size    = 4
    seq_len       = 128
    vocab_size    = 1024
    num_vq_layers = 4

    model = TokenTransformerMLM(
        vocab_size=vocab_size,
        d_model=128,          # small for fast testing
        nhead=4,
        num_layers=2,
        max_seq_len=256,
        num_vq_layers=num_vq_layers,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"MASK_TOKEN_ID : {model.MASK_TOKEN_ID}")
    print(f"PAD_TOKEN_ID  : {model.PAD_TOKEN_ID}\n")

    # ── test 1: basic forward pass ────────────────────────────────────────────
    print("── Test 1: basic forward pass ──")
    input_ids = torch.randint(0, vocab_size, (batch_size, num_vq_layers, seq_len))
    with torch.no_grad():
        logits_list = model(input_ids)

    assert len(logits_list) == num_vq_layers, \
        f"Expected {num_vq_layers} output tensors, got {len(logits_list)}"
    for q, logits_q in enumerate(logits_list):
        assert logits_q.shape == (batch_size, seq_len, vocab_size), \
            f"Layer {q+1}: expected {(batch_size, seq_len, vocab_size)}, got {logits_q.shape}"
        print(f"  L{q+1} logits shape: {logits_q.shape}  ✅")

    # ── test 2: forward with padding mask ─────────────────────────────────────
    print("\n── Test 2: forward with padding mask ──")
    pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    pad_mask[:, -10:] = True    # last 10 positions are padding
    with torch.no_grad():
        logits_list_padded = model(input_ids, src_key_padding_mask=pad_mask)
    print(f"  Forward with pad_mask: OK  ✅")

    # ── test 3: MLM loss ──────────────────────────────────────────────────────
    print("\n── Test 3: MLM loss computation ──")
    # simulate masking: ~15% of positions
    mask_prob  = 0.15
    labels     = input_ids.clone()
    mask_flags = torch.rand(batch_size, seq_len) < mask_prob    # [B, L]

    # replace masked positions with MASK token (all 4 layers simultaneously)
    masked_input = input_ids.clone()
    for q in range(num_vq_layers):
        masked_input[:, q, :][mask_flags] = model.MASK_TOKEN_ID

    # set non-masked positions to -100 (ignore in loss)
    for q in range(num_vq_layers):
        labels[:, q, :][~mask_flags] = -100

    masked_input = masked_input
    with torch.no_grad():
        logits_list_masked = model(masked_input)
    loss = model.compute_mlm_loss(logits_list_masked, labels)
    print(f"  MLM loss: {loss.item():.4f}  ✅")
    print(f"  Masked positions: {mask_flags.sum().item()} / {batch_size * seq_len}")

    # ── test 4: wrong num_vq_layers raises AssertionError ─────────────────────
    print("\n── Test 4: wrong input shape raises assertion ──")
    bad_input = torch.randint(0, vocab_size, (batch_size, 3, seq_len))  # wrong: 3 layers
    try:
        model(bad_input)
        print("  ❌ Should have raised AssertionError")
    except AssertionError as e:
        print(f"  AssertionError raised correctly: {e}  ✅")

    print("\n✅ All tests passed.")
