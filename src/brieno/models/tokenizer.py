import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ==========================================
# K-means utilities (for codebook initialization)
# https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py
# ==========================================
def sample_vectors(samples, num):
    """Sample `num` vectors from `samples` along dim 0.

    Args:
        samples: [N, D] tensor.
        num:     number of vectors to draw.

    Returns:
        [num, D] tensor. Samples without replacement if N >= num,
        otherwise samples with replacement.
    """
    num_samples = samples.shape[0]
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=samples.device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=samples.device)
    return samples[indices]


def kmeans(samples, num_clusters, num_iters=50):
    """Standard Lloyd k-means.

    Args:
        samples:      [N, D] data points.
        num_clusters: K, number of clusters / codebook entries.
        num_iters:    number of Lloyd iterations.

    Returns:
        means: [K, D] cluster centroids.
        bins:  [K]    final assignment counts (long tensor).
    """
    dim = samples.shape[-1]
    # Random init from the data itself
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # Pairwise L2 distance [N, K]; cdist is memory-efficient vs. broadcasting
        dists = torch.cdist(samples.unsqueeze(0), means.unsqueeze(0)).squeeze(0)
        buckets = dists.argmin(dim=-1)  # [N]

        # Count assignments per cluster
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = (bins == 0)
        bins_clamped = bins.masked_fill(zero_mask, 1)

        # Recompute centroids
        new_means = torch.zeros_like(means)
        new_means.scatter_add_(
            0, buckets.unsqueeze(-1).expand(-1, dim), samples
        )
        new_means = new_means / bins_clamped.unsqueeze(-1).to(new_means.dtype)

        # Keep old centroids for empty clusters (avoid drifting to 0 / NaN)
        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


# ==========================================
# Basic Module
# ==========================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=3, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.conv(x))))


class ConvTransposeBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1, dropout=0.0):
        super().__init__()
        # Upsampling
        self.upconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.upconv(x))))


# ==========================================
# Downsample factor -> number of stride-2 levels (Group 3: 4x / 8x / 16x)
# ==========================================
SUPPORTED_DOWNSAMPLE = (4, 8, 16)


def downsample_levels(downsample: int) -> int:
    ds = int(downsample)
    if ds not in SUPPORTED_DOWNSAMPLE:
        raise ValueError(f"downsample must be one of {SUPPORTED_DOWNSAMPLE}, got {downsample}")
    return int(round(math.log2(ds)))


# ==========================================
# Encoder: 16x downsample (default); 8x / 4x = the same stem with the last level(s) removed
# ==========================================
class SignalEncoder(nn.Module):
    """Conv stem of `log2(downsample)` stride-2 blocks + a 3x3 projection to hidden_dim.

    Channel ladder hidden/4 -> hidden/2 -> hidden -> hidden (first block k=7, others k=5).
    With downsample=16 (default) the module names and shapes are IDENTICAL to the original
    fixed 16x encoder, so every existing checkpoint loads with strict=True.
    8x keeps the first three blocks, 4x the first two; the projection then maps the last
    block's width to hidden_dim so the latent is always [B, hidden_dim, N/downsample].
    """
    def __init__(self, in_channels=1, hidden_dim=256, dropout=0.1, downsample=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.downsample = int(downsample)
        levels = downsample_levels(self.downsample)

        ladder = [hidden_dim // 4, hidden_dim // 2, hidden_dim, hidden_dim][:levels]
        blocks, c_in = [], in_channels
        for i, c_out in enumerate(ladder):
            k, p = (7, 3) if i == 0 else (5, 2)
            blocks.append(ConvBlock1D(c_in, c_out, kernel_size=k, stride=2, padding=p, dropout=dropout))
            c_in = c_out
        self.stem = nn.Sequential(*blocks)

        self.final_proj = nn.Sequential(
            nn.Conv1d(c_in, hidden_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.stem(x)
        return self.final_proj(x)


# ==========================================
# Decoder: mirror of the encoder (16x default)
# ==========================================
class SignalDecoder(nn.Module):
    """`log2(downsample) - 1` ConvTranspose blocks halving the width each time, then a final
    ConvTranspose to `out_channels`. downsample=16 reproduces the original module names/shapes
    exactly (hidden -> /2 -> /4 -> /8 -> out)."""
    def __init__(self, hidden_dim=256, out_channels=1, dropout=0.1, downsample=16):
        super().__init__()
        self.downsample = int(downsample)
        levels = downsample_levels(self.downsample)

        self.initial_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        blocks, c = [], hidden_dim
        for _ in range(levels - 1):
            blocks.append(ConvTransposeBlock1D(c, c // 2, kernel_size=5, stride=2, padding=2,
                                               output_padding=1, dropout=dropout))
            c = c // 2
        blocks.append(nn.ConvTranspose1d(c, out_channels, kernel_size=7, stride=2, padding=3,
                                         output_padding=1))    # last level outputs the signal, no dropout
        self.up_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.initial_proj(x)
        return self.up_blocks(x)


# ==========================================
# Single Codebook
# ==========================================
class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_tokens, embedding_dim, commitment_cost, ema_decay=0.99,
                 cluster_epsilon=1e-5, noise_epsilon=1e-4,
                 kmeans_init=True, kmeans_iters=50):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # EMA update parameters
        self.ema_decay = ema_decay
        self.cluster_epsilon = cluster_epsilon
        self.noise_epsilon = noise_epsilon

        # K-means initialization parameters
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        embed = torch.randn(num_tokens, embedding_dim) * 0.01

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_tokens))   # EMA Hit Count
        self.register_buffer("embed_avg", embed.clone())
        # inited[0] = True iff codebook has been initialized.
        # If kmeans_init is False, treat as already initialized.
        self.register_buffer("inited", torch.tensor([not kmeans_init], dtype=torch.bool))
        self.register_buffer("init_buffer", torch.empty(0, embedding_dim))
        self.init_buffer_size = 25600

    @torch.no_grad()
    def init_embed_(self, flat_x):
        """Initialize codebook from k-means on the first training batch.

        Args:
            flat_x: [N, D] tensor of encoder outputs (already detached).

        DDP: only rank 0 runs k-means; result is broadcast so every rank
        ends up with identical centroids.
        """
        if self.inited.item():
            return

        self.init_buffer = torch.cat([self.init_buffer, flat_x], dim=0)
        if self.init_buffer.shape[0] < self.init_buffer_size:
            return

        # reach the self.init_buffer_size, start to cluster using kmeans
        device = flat_x.device
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            if dist.get_rank() == 0:
                embed, bins = kmeans(self.init_buffer, self.num_tokens, self.kmeans_iters)
            else:
                embed = torch.zeros(self.num_tokens, self.embedding_dim, device=device)
                bins = torch.zeros(self.num_tokens, device=device, dtype=torch.long)
            dist.broadcast(embed, src=0)
            dist.broadcast(bins, src=0)
        else:
            embed, bins = kmeans(self.init_buffer, self.num_tokens, self.kmeans_iters)

        cluster_size = bins.to(self.cluster_size.dtype)
        # embed_avg tracks (sum of assigned vectors) per code for EMA consistency.
        # Since means = sum / count, set embed_avg = means * count so the very
        # first EMA update mixes correctly-scaled quantities.
        embed_avg = embed * cluster_size.unsqueeze(1)

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_avg)
        self.cluster_size.data.copy_(cluster_size)

        self.inited.data.fill_(True)
        # self.init_buffer = torch.empty(0, self.embedding_dim, device=device)
        self.init_buffer.data = torch.empty(0, self.embedding_dim, device=device)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()       # x shape: [B, D, L] -> [B, L, D]
        flat_x = x.view(-1, self.embedding_dim)   # flat_x shape: [B*L, D]

        # ------------------- K-means initialization  -------------------
        if self.training and not self.inited.item():
            self.init_embed_(flat_x.detach())
        # ---------------------------------------------------------------

        #  (x - y)^2 = x^2 - 2xy + y^2
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embed ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embed.t()))

        # find the index of codebook
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_tokens, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embed)
        quantized = quantized.view(x.shape)

        # update codebook using EMA during training
        if self.training and self.inited.item():
            encodings_sum = encodings.sum(0).detach()   #  hits count
            embed_sum = torch.matmul(encodings.t(), flat_x.detach())

            #  DDP
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(encodings_sum)
                dist.all_reduce(embed_sum)

            # update cluster size
            self.cluster_size.data.mul_(self.ema_decay).add_(encodings_sum, alpha=1 - self.ema_decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.cluster_epsilon)
                    / (n + self.num_tokens * self.cluster_epsilon) * n
            )
            # update embed sum and embed
            self.embed_avg.data.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            new_embed = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(new_embed)

            # ------------------- Dead Code Restart -------------------
            usage = (self.cluster_size >= 0.5)

            # usage may differ at different GPUs, add 2026.04.11
            if dist.is_available() and dist.is_initialized():
                usage_int = usage.int()
                dist.all_reduce(usage_int, op=dist.ReduceOp.MIN)
                usage = usage_int.bool()

            if not usage.all():
                # find the index of dead codes
                dead_codes = torch.nonzero(~usage).squeeze(1)
                num_dead = len(dead_codes)

                # only use the GPU 0 for sampling, because flat_x maybe different on other GPUs
                # if drop_last=True in DataLoader, the following is not necessary
                if dist.is_available() and dist.is_initialized():
                    max_samples_tensor = torch.tensor([flat_x.shape[0]], device=flat_x.device)
                    dist.broadcast(max_samples_tensor, src=0)
                    max_samples = max_samples_tensor.item()
                else:
                    max_samples = flat_x.shape[0]

                num_samples = min(num_dead, max_samples)
                if num_samples < num_dead:
                    dead_codes = dead_codes[:num_samples]

                sampled_active_features = torch.zeros(
                    num_samples, self.embedding_dim, device=flat_x.device
                )

                # replace the dead code using the sample from flat_x of GPU 0
                if (not dist.is_initialized()) or \
                        (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
                    random_indices = torch.randperm(flat_x.shape[0], device=flat_x.device)[:num_samples]
                    sampled_features = flat_x.detach()[random_indices]
                    noise = torch.randn_like(sampled_features) * self.noise_epsilon
                    sampled_features = sampled_features + noise
                    sampled_active_features.copy_(sampled_features)

                # DDP
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(sampled_active_features, src=0)

                # replace the dead codebook, reset the cluster_size (EMA Hit Count)
                self.embed.data[dead_codes] = sampled_active_features.clone()
                self.cluster_size.data[dead_codes] = 1.0
                self.embed_avg.data[dead_codes] = sampled_active_features.clone()
            # -----------------------------------------------

        # Commitment Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        # https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py#L301
        quantized = x + (quantized - x).detach()

        # [B, D, L]
        return quantized.permute(0, 2, 1), encoding_indices.view(x.shape[0], x.shape[1]), loss


# ==========================================
# RVQ
# ==========================================
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=4, num_tokens=1024, embedding_dim=256,
                 commitment_cost=0.25, ema_decay=0.99,
                 kmeans_init=True, kmeans_iters=50):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.vqs = nn.ModuleList([
            EMAVectorQuantizer(num_tokens=num_tokens,
                               embedding_dim=embedding_dim,
                               commitment_cost=commitment_cost,
                               ema_decay=ema_decay,
                               kmeans_init=kmeans_init,
                               kmeans_iters=kmeans_iters,
                               )
            for _ in range(num_quantizers)
        ])

    def forward(self, x, return_layers: bool = False):
        """return_layers=False (default): (quantized_sum, indices [Q,B,L], total_loss, losses) -- unchanged API.
        return_layers=True: additionally returns the list of per-layer quantized outputs (each [B,D,L], straight-through,
        so gradients reach the encoder through any single layer), used by the guide heads when guide_layers != 'all'."""
        quantized_out = torch.zeros_like(x)
        residual = x

        all_indices = []
        all_losses = []  # save loss for each vq
        layer_quantized = []
        total_loss = torch.tensor(0.0, device=x.device)

        # https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py#L338
        # NOTE: each VQ's init_embed_() runs on the residual seen by THAT layer,
        # which is exactly the right distribution for its k-means init.
        for vq in self.vqs:
            quantized, indices, loss = vq(residual)
            quantized_out = quantized_out + quantized
            residual = residual - quantized

            all_indices.append(indices)
            all_losses.append(loss)
            layer_quantized.append(quantized)
            total_loss += loss

        # all_indices shape: [num_quantizers, B, L]
        if return_layers:
            return quantized_out, torch.stack(all_indices), total_loss, all_losses, layer_quantized
        return quantized_out, torch.stack(all_indices), total_loss, all_losses


# ==========================================
# Grouped FSQ (Group 5: finite scalar quantization, Mentzer et al. 2023)
# ==========================================
class GroupedFSQ(nn.Module):
    """`num_groups` independent FSQ codes per frame (drop-in for ResidualVectorQuantizer).

    Input dim = num_groups * len(levels). Each group of len(levels) scalars is bounded with
    tanh to [-(L-1)/2, (L-1)/2] per dimension (half-integer offset for even L), rounded with a
    straight-through estimator, and the per-dimension integers are combined into ONE code in
    [0, prod(levels)). No codebook, no EMA, no commitment loss: `total_loss` is a zero tensor.

    Output contract matches the RVQ: (quantized [B,D,L] in [-1,1], indices [G,B,L], total_loss,
    losses_list). `losses_list[g]` = mean squared rounding error of group g (diagnostic only,
    detached). With return_layers=True the extra list holds the quantized tensor masked to each
    group's dimensions (so their sum equals `quantized`).
    """
    def __init__(self, num_groups=4, levels=(8, 5, 5, 5)):
        super().__init__()
        self.levels = [int(l) for l in levels]
        assert all(l >= 2 for l in self.levels), self.levels
        self.num_groups = int(num_groups)
        self.num_quantizers = self.num_groups           # API parity with ResidualVectorQuantizer
        self.dim_per_group = len(self.levels)
        self.embedding_dim = self.num_groups * self.dim_per_group
        self.num_tokens = int(math.prod(self.levels))

        lv = torch.tensor(self.levels, dtype=torch.float32)
        basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.int64), dim=0)
        self.register_buffer("_levels", lv)
        self.register_buffer("_basis", basis)

    def _bound(self, z, eps: float = 1e-3):
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, torch.tensor(0.5, device=z.device), torch.tensor(0.0, device=z.device))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, x, return_layers: bool = False):
        B, D, L = x.shape
        assert D == self.embedding_dim, f"GroupedFSQ expects {self.embedding_dim} channels, got {D}"
        z = x.permute(0, 2, 1).reshape(B, L, self.num_groups, self.dim_per_group)   # [B,L,G,k]
        zb = self._bound(z)
        zq = zb + (torch.round(zb) - zb).detach()                                      # STE
        half_width = (self._levels // 2).to(zq.dtype)
        codes = (torch.round(zb) + half_width).long()                                  # 0..L_i-1 per dim
        indices = (codes * self._basis).sum(-1)                                        # [B,L,G]
        q_norm = zq / half_width                                                       # [-1,1]

        quantized = q_norm.reshape(B, L, D).permute(0, 2, 1)                           # [B,D,L]
        all_indices = indices.permute(2, 0, 1).contiguous()                            # [G,B,L]
        losses = [((zb[:, :, g] - torch.round(zb[:, :, g])) ** 2).mean().detach() for g in range(self.num_groups)]
        total_loss = x.new_zeros(())
        if return_layers:
            layer_quantized = []
            for g in range(self.num_groups):
                m = torch.zeros(self.num_groups, self.dim_per_group, device=x.device, dtype=quantized.dtype)
                m[g] = 1.0
                layer_quantized.append((q_norm * m).reshape(B, L, D).permute(0, 2, 1))
            return quantized, all_indices, total_loss, losses, layer_quantized
        return quantized, all_indices, total_loss, losses


# ==========================================
# Tokenizer Pretraining
# ==========================================
class SignalTokenizer(nn.Module):
    """Encoder -> (pre_quant) -> quantizer -> (post_quant) -> Decoder.

    New optional structure knobs (2026-10-08; all defaults reproduce the original model exactly,
    including state_dict keys, so every existing checkpoint still loads with strict=True):
      downsample : 16 (default) | 8 | 4      encoder stride / token rate   (Group 3)
      quant_dim  : None (default) | int      quantize in a low-dim projected space; adds
                                             pre_quant / post_quant 1x1 convs           (Group 5)
      quantizer  : "rvq" (default) | "fsq"   residual VQ or grouped finite scalar quantization
      fsq_levels : e.g. [8,5,5,5]            per-dimension levels for FSQ; the quantized dim is
                                             num_quantizers * len(levels) (projection implied) and
                                             num_tokens must equal prod(levels)
    Use `quantize(latent)` instead of calling `self.quantizer` directly when a projection may be
    present; for the default configuration both are numerically identical.
    """
    def __init__(self, in_channels=1, hidden_dim=256, num_quantizers=4, num_tokens=1024,
                 commitment_cost=0.25, ema_decay=0.99, encoder_dropout=0.1, decoder_dropout=0.0,
                 kmeans_init=True, kmeans_iters=50,
                 downsample=16, quant_dim=None, quantizer="rvq", fsq_levels=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.downsample = int(downsample)
        self.num_quantizers = int(num_quantizers)
        self.quantizer_type = str(quantizer)

        self.encoder = SignalEncoder(in_channels, hidden_dim, encoder_dropout, downsample=self.downsample)

        if self.quantizer_type == "fsq":
            levels = [int(l) for l in (fsq_levels or (8, 5, 5, 5))]
            qd = self.num_quantizers * len(levels)
            if quant_dim is not None and int(quant_dim) != qd:
                raise ValueError(f"fsq: quant_dim must be num_quantizers*len(levels)={qd}, got {quant_dim}")
            if int(num_tokens) != math.prod(levels):
                raise ValueError(f"fsq: num_tokens must equal prod(fsq_levels)={math.prod(levels)}, got {num_tokens}")
            self.quant_dim = qd
            self.quantizer = GroupedFSQ(self.num_quantizers, levels)
        elif self.quantizer_type == "rvq":
            self.quant_dim = int(quant_dim) if quant_dim is not None else None
            emb_dim = self.quant_dim if self.quant_dim is not None else hidden_dim
            self.quantizer = ResidualVectorQuantizer(
                num_quantizers=num_quantizers,
                num_tokens=num_tokens,
                embedding_dim=emb_dim,
                commitment_cost=commitment_cost,
                ema_decay=ema_decay,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters)
        else:
            raise ValueError(f"quantizer must be 'rvq' or 'fsq', got {quantizer!r}")
        self.num_tokens = int(num_tokens)

        # Only created when quantizing in a projected space -> absent (and no state_dict keys) otherwise.
        if self.quant_dim is not None and self.quant_dim != hidden_dim:
            self.pre_quant = nn.Conv1d(hidden_dim, self.quant_dim, kernel_size=1)
            self.post_quant = nn.Conv1d(self.quant_dim, hidden_dim, kernel_size=1)

        self.decoder = SignalDecoder(hidden_dim, in_channels, decoder_dropout, downsample=self.downsample)

    # ---------------------------------------------------------------- helpers
    @property
    def has_projection(self) -> bool:
        return hasattr(self, "pre_quant")

    def latent_len(self, n_samples: int) -> int:
        return int(n_samples) // self.downsample

    @classmethod
    def from_config(cls, m: dict, **overrides):
        """Build from the `model` section of an experiment_config.json (legacy keys honoured:
        a single `dropout` falls back for both encoder/decoder dropout). `overrides` win, e.g.
        kmeans_init=False for evaluation or commitment_cost=1.0 to read VQ_MSE directly."""
        legacy = float(m.get("dropout", 0.0))
        kw = dict(
            in_channels=1,
            hidden_dim=int(m["hidden_dim"]),
            num_quantizers=int(m["num_quantizers"]),
            num_tokens=int(m["num_tokens"]),
            commitment_cost=float(m["commitment_cost"]),
            ema_decay=float(m["ema_decay"]),
            encoder_dropout=float(m.get("encoder_dropout", legacy)),
            decoder_dropout=float(m.get("decoder_dropout", legacy)),
            kmeans_init=True,
            downsample=int(m.get("downsample", 16)),
            quant_dim=m.get("quant_dim", None),
            quantizer=m.get("quantizer", "rvq"),
            fsq_levels=m.get("fsq_levels", None),
        )
        kw.update(overrides)
        return cls(**kw)

    @staticmethod
    def config_tag(m: dict) -> str:
        """Suffix for run UIDs encoding the non-default structure knobs, e.g. '_ds8_qd64' or
        '_ds8_fsq8x5x5x5'. Empty for the legacy configuration so existing UIDs are unchanged."""
        parts = []
        ds = int(m.get("downsample", 16))
        if ds != 16:
            parts.append(f"ds{ds}")
        qz = m.get("quantizer", "rvq")
        if qz == "fsq":
            parts.append("fsq" + "x".join(str(int(l)) for l in (m.get("fsq_levels") or (8, 5, 5, 5))))
        elif m.get("quant_dim") is not None:
            parts.append(f"qd{int(m['quant_dim'])}")
        nt = int(m.get("num_tokens", 1024))
        if nt != 1024 and qz != "fsq":
            parts.append(f"nt{nt}")
        return ("_" + "_".join(parts)) if parts else ""

    # ---------------------------------------------------------------- core
    def quantize(self, latent, return_layers: bool = False):
        """latent [B, hidden, L] (encoder output) -> (quantized [B, hidden, L], tokens [Q,B,L], loss, losses[, layer_q]).
        Runs in fp32 with autocast disabled (as the original forward did). With a projection the
        per-layer tensors in `layer_q` live in the quantized (low-dim) space."""
        with torch.cuda.amp.autocast(enabled=False):
            z = latent.float()
            if self.has_projection:
                z = self.pre_quant(z)
            out = self.quantizer(z, return_layers=True) if return_layers else self.quantizer(z)
            q = out[0]
            if self.has_projection:
                q = self.post_quant(q)
        if return_layers:
            return q, out[1], out[2], out[3], out[4]
        return q, out[1], out[2], out[3]

    def forward(self, x):
        """
            Input:
                x: [Batch, 1, 16384]  (Raw waveform)
            Output:
                reconstructed_x: [Batch, 1, 16384]  (Reconstructed waveform)
                token_indices: [num_quantizers, Batch, seq_len = 16384 / downsample]
                               *Note: Values inside range from 0 to num_tokens-1
                vq_total_loss: Scalar  (Commitment loss; 0 for FSQ)
                vq_losses_list: List
            """
        latent = self.encoder(x)
        quantized_latent, token_indices, vq_total_loss, vq_losses_list = self.quantize(latent)
        reconstructed_x = self.decoder(quantized_latent)
        return reconstructed_x, token_indices, vq_total_loss, vq_losses_list
