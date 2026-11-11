"""Transformer pretraining: masked-token prediction over tokenized signal.

Spans of token positions are hidden and the model predicts them from their context,
in every quantizer layer at once. This is the stage that turns a per-chunk code into a
contextual representation, and it is what the downstream heads are attached to.

The target is always the code ids of the hidden frames. What the model reads is set by
``model.input_mode`` in the config:

``token`` (default)
    The model reads code ids. The corpus is tokenized once with ``brieno train encode``
    and training never touches the tokenizer again.

``latent``
    The model reads the tokenizer's continuous latent and predicts the code ids the
    quantizer would have assigned. The corpus is the *signal* corpus — a stored latent
    would be a hundred times its size — and the frozen tokenizer runs during training
    to produce both the input and the targets.

Run on one GPU::

    brieno train transformer -c config.json

or on several with torchrun::

    torchrun --nproc_per_node=4 -m brieno.cli train transformer -c config.json
"""
import datetime
import json
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from brieno.models.transformer import TokenTransformerMLM
from brieno.train.data import (BlendedTokenDataset, MergedSignalDataset,
                               SignalSpanMaskCollator, SpanMaskCollator, TokenDataset)
from brieno.train.utils import (
    JsonlLogger,
    autocast,
    cleanup_distributed,
    cosine_schedule_with_warmup,
    grad_scaler,
    setup_distributed,
    worker_init_fn,
)

__all__ = ["layer_accuracy", "pretrain_transformer"]


def layer_accuracy(logits_list, labels, vocab_size, layer_weights):
    """Top-1 and top-5 accuracy at masked positions, per quantizer layer.

    Layers with zero loss weight are skipped: they are not being trained, so their
    accuracy is noise and computing it on every batch is not free.
    """
    out = {}
    for q, (logits_q, weight) in enumerate(zip(logits_list, layer_weights)):
        key = f"L{q + 1}"
        labels_q = labels[:, q, :].reshape(-1)
        valid = labels_q != -100
        if weight == 0.0 or valid.sum() == 0:
            out[key] = (0.0, 0.0)
            continue
        picked = logits_q.reshape(-1, vocab_size)[valid]
        truth = labels_q[valid]
        top1 = (picked.argmax(-1) == truth).float().mean().item()
        top5 = (picked.topk(5, dim=-1).indices == truth.unsqueeze(1)).any(-1).float().mean().item()
        out[key] = (top1, top5)
    return out


def _build(cfg, mode, world_size, verbose):
    """Training blend (mode='train') or concatenated validation splits (mode='val').

    The files under ``datasets.root_dir`` are token arrays in token mode and signal
    arrays in latent mode. Both are split by the same seed, so a dataset's held-out
    chunks are the same chunks either way.
    """
    root = cfg["datasets"]["root_dir"]
    val_size = cfg["datasets"]["validation_size_per_dataset"]
    Corpus = (MergedSignalDataset if cfg["model"].get("input_mode", "token") == "latent"
              else TokenDataset)

    if mode == "val":
        parts = []
        for name in cfg["datasets"]["validation"]:
            ds = Corpus(os.path.join(root, name + ".npy"), mode="val", val_size=val_size)
            parts.append(ds)
            if verbose:
                print(f"  [val]   {name:50s} chunks={len(ds):>10,d}")
        return ConcatDataset(parts)

    datasets, weights = [], []
    for entry in cfg["datasets"]["weights"]:
        ds = Corpus(os.path.join(root, entry["name"] + ".npy"),
                    mode="train", val_size=val_size)
        datasets.append(ds)
        weights.append(float(entry["weight"]))
        if verbose:
            print(f"  [train] {entry['name']:50s} chunks={len(ds):>10,d}  weight={entry['weight']}")

    total = sum(weights)
    weights = [w / total for w in weights]
    return BlendedTokenDataset(datasets, weights,
                               epoch_length=cfg["training"]["epoch_length"],
                               batch_size=cfg["training"]["batch_size"],
                               world_size=world_size)


def _load_frozen_tokenizer(cfg, device):
    """Latent mode: load the tokenizer that supplies the input and the targets, and
    make the transformer's config agree with it.

    The transformer's vocabulary, codebook count and input width are properties of the
    tokenizer, not free choices; taking them from the tokenizer — and refusing a config
    that contradicts it — is what keeps a mismatch from surfacing as a shape error
    deep in the first batch, or worse, not at all.
    """
    from brieno.data.encode import load_tokenizer

    path = cfg.get("tokenizer")
    if not path:
        raise ValueError("model.input_mode is 'latent': a tokenizer checkpoint is needed "
                         "to compute the input and the targets (--tokenizer, or a "
                         "top-level \"tokenizer\" entry in the config)")
    tok, tok_cfg = load_tokenizer(path, device=device)
    for p in tok.parameters():
        p.requires_grad_(False)

    m = cfg["model"]
    for key, expected, what in (
            ("vocab_size", int(tok_cfg["num_tokens"]), "codebook size"),
            ("num_vq_layers", int(tok_cfg["num_quantizers"]), "number of codebooks")):
        if int(m.get(key, expected)) != expected:
            raise ValueError(f"model.{key} is {m[key]} but the tokenizer's {what} is "
                             f"{expected}; the targets are the tokenizer's code ids")
        m[key] = expected
    if m.get("latent_dim") not in (None, tok.latent_dim):
        raise ValueError(f"model.latent_dim is {m['latent_dim']} but the tokenizer's "
                         f"latent is {tok.latent_dim}-d")
    m["latent_dim"] = tok.latent_dim
    return tok


def _prepare(batch, device, tokenizer=None):
    """A collated batch -> ``(model input, mask, labels)`` for either input mode.

    Token mode: the collator already wrote the mask into the ids, so the batch passes
    through. Latent mode: the frozen tokenizer turns the signal into the latent the
    model reads and the code ids it must predict, and the collator's masking decision
    is applied to the latent here.
    """
    if tokenizer is None:
        input_ids, labels = batch
        return (input_ids.to(device, non_blocking=True), None,
                labels.to(device, non_blocking=True))

    signal, masked, replace, random, source = (t.to(device, non_blocking=True) for t in batch)
    with torch.no_grad():
        # fp32 on purpose, outside the trainer's autocast: a code id is a
        # nearest-neighbour decision, and half precision moves the boundaries. These
        # targets are then the ids `brieno train encode` writes for the same chunks.
        latent, tokens = tokenizer.encode(signal)
    latent = latent.float()
    labels = tokens.masked_fill(~masked.unsqueeze(1), -100)
    if random.any():
        donors = latent.reshape(-1, latent.shape[-1])[source[random]]
        latent = latent.clone()
        latent[random] = donors
    return latent, replace, labels


def _save(path, model, optimizer, scheduler, scaler, cfg, epoch, global_step, is_distributed):
    torch.save({
        "format": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": cfg["model"],
        **({"tokenizer": os.path.basename(cfg["tokenizer"])} if cfg.get("tokenizer") else {}),
        "model_state_dict": (model.module if is_distributed else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)


def pretrain_transformer(config_path, seed=None, run_name=None, resume=None, overrides=None,
                         tokenizer=None):
    """Pretrain a :class:`TokenTransformerMLM` from a JSON config. Returns the output directory.

    ``tokenizer`` is a tokenizer checkpoint, needed in latent mode only; it overrides
    the config's top-level ``tokenizer`` entry.
    """
    is_distributed, local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = global_rank == 0

    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for section, key, value in (overrides or []):
        if value is not None:
            cfg[section][key] = value
    if tokenizer is not None:
        cfg["tokenizer"] = tokenizer

    input_mode = cfg["model"].get("input_mode", "token")
    frozen_tokenizer = None
    if input_mode == "latent":
        frozen_tokenizer = _load_frozen_tokenizer(cfg, device)

    seed = int(cfg["training"].get("seed", 42)) if seed is None else int(seed)
    torch.manual_seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    random.seed(seed + global_rank)

    if is_main:
        name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(cfg["training"]["save_dir"], name)
        os.makedirs(out_dir, exist_ok=True)
        cfg["training"]["save_dir"] = out_dir
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"writing to {out_dir}")

    if is_distributed:
        dist.barrier()
        box = [cfg]
        dist.broadcast_object_list(box, src=0)
        cfg = box[0]
        dist.barrier()

    out_dir = cfg["training"]["save_dir"]
    history = JsonlLogger(os.path.join(out_dir, "history.jsonl"), enabled=is_main)

    vocab_size = int(cfg["model"]["vocab_size"])
    n_vq = int(cfg["model"].get("num_vq_layers", 4))
    layer_weights = cfg["masking"].get("layer_weights", [1.0] * n_vq)
    grad_accum = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    batch_size = int(cfg["training"]["batch_size"])

    # ---------------------------------------------------------------- data
    if is_main:
        print("\nbuilding datasets ...")
    train_set = _build(cfg, "train", world_size, is_main)
    val_set = _build(cfg, "val", world_size, is_main)

    # The blend is sized for `epoch_length` optimizer steps, so it must hold
    # grad_accum times as many micro-batches.
    if grad_accum > 1:
        train_set.epoch_samples *= grad_accum

    if frozen_tokenizer is None:
        collate = SpanMaskCollator(mask_ratio=cfg["masking"]["mask_ratio"],
                                   min_span=cfg["masking"]["min_span"],
                                   mask_token_id=vocab_size,      # == model.MASK_TOKEN_ID
                                   vocab_size=vocab_size)
    else:
        collate = SignalSpanMaskCollator(mask_ratio=cfg["masking"]["mask_ratio"],
                                         min_span=cfg["masking"]["min_span"],
                                         downsample=frozen_tokenizer.downsample)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank,
                                       shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=global_rank,
                                     shuffle=False) if is_distributed else None

    workers = int(cfg["training"].get("num_workers", 4))
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              collate_fn=collate, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                            num_workers=workers, pin_memory=True, drop_last=False,
                            collate_fn=collate)

    # ---------------------------------------------------------------- model
    model = TokenTransformerMLM.from_config(cfg["model"]).to(device)
    base_model = model
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\ntransformer: {cfg['model']['n_layer']} layers, {cfg['model']['n_head']} heads, "
              f"d_model {cfg['model']['d_model']}, {n_vq} quantizer layers "
              f"| {n_params / 1e6:.1f} M parameters")
        print("input: code ids" if frozen_tokenizer is None else
              f"input: continuous latent ({cfg['model']['latent_dim']}-d) from the frozen "
              f"tokenizer {os.path.basename(cfg['tokenizer'])}; targets: its code ids")
        print(f"batch {batch_size} x world {world_size} x accum {grad_accum} "
              f"= effective {batch_size * world_size * grad_accum}")

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        base_model = model.module

    # Biases are excluded from weight decay. Everything else is decayed, LayerNorm
    # gains and the positional table included: this is the grouping the released
    # weights were trained with, and it is kept so that the recipe stays theirs.
    no_decay = ("bias",)
    decayed, plain = [], []
    for name, param in model.named_parameters():
        (plain if any(tag in name for tag in no_decay) else decayed).append(param)
    # betas / eps likewise follow the transformer convention rather than PyTorch's
    # AdamW defaults (0.999 / 1e-8).
    optimizer = optim.AdamW(
        [{"params": decayed, "weight_decay": cfg["training"]["weight_decay"]},
         {"params": plain, "weight_decay": 0.0}],
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"].get("betas", (0.9, 0.98))),
        eps=float(cfg["training"].get("eps", 1e-6)),
    )
    if is_main:
        print(f"optimizer: AdamW, {len(decayed)} tensors decayed, "
              f"{len(plain)} exempt (biases)")
    epochs = int(cfg["training"]["epochs"])
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg["training"].get("warmup_steps", int(total_steps * 0.05)))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = grad_scaler()

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if resume:
        state = torch.load(resume, map_location=device)
        base_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        if is_main:
            print(f"resumed from {resume} at epoch {state['epoch']}")

    log_every = int(cfg.get("logging", {}).get("log_every_n_steps", 20))

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        running, n_micro_seen, t0 = 0.0, 0, time.time()
        # Drop the tail that cannot fill an accumulation group, so every optimizer
        # step averages the same number of micro-batches.
        n_micro = len(train_loader) - (len(train_loader) % grad_accum)

        for step, batch in enumerate(train_loader):
            if step >= n_micro:
                break
            x, mask, labels = _prepare(batch, device, frozen_tokenizer)

            is_last_micro = (step + 1) % grad_accum == 0
            # Skip the DDP all-reduce on every micro-batch but the last of a group.
            sync = nullcontext() if (is_last_micro or not is_distributed) else model.no_sync()
            with sync:
                with autocast():
                    logits_list = model(x, mask=mask)
                    loss = base_model.compute_mlm_loss(logits_list, labels, layer_weights)
                scaler.scale(loss / grad_accum).backward()

            running += loss.item()
            n_micro_seen += 1

            if not is_last_micro:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main and global_step % log_every == 0:
                with torch.no_grad():
                    acc = layer_accuracy(logits_list, labels, vocab_size, layer_weights)
                history.log(split="train", epoch=epoch, step=global_step, loss=loss.item(),
                            lr=optimizer.param_groups[0]["lr"],
                            **{f"top1_{k}": v[0] for k, v in acc.items()},
                            **{f"top5_{k}": v[1] for k, v in acc.items()})
                print(f"  epoch {epoch}/{epochs}  step {global_step}  loss {loss.item():.4f}  "
                      f"L1 top1 {acc['L1'][0]:.3f}  lr {optimizer.param_groups[0]['lr']:.2e}",
                      flush=True)

        if is_main:
            print(f"epoch {epoch} train | loss {running / max(n_micro_seen, 1):.4f} "
                  f"({time.time() - t0:.0f}s)")

        # ------------------------------------------------------------ validation
        model.eval()
        vloss, vbatches = 0.0, 0
        acc_sum = {f"L{q + 1}": [0.0, 0.0] for q in range(n_vq)}

        with torch.no_grad():
            for batch in val_loader:
                x, mask, labels = _prepare(batch, device, frozen_tokenizer)
                with autocast():
                    logits_list = model(x, mask=mask)
                    loss = base_model.compute_mlm_loss(logits_list, labels, layer_weights)
                acc = layer_accuracy(logits_list, labels, vocab_size, layer_weights)
                for k, (t1, t5) in acc.items():
                    acc_sum[k][0] += t1
                    acc_sum[k][1] += t5
                vloss += loss.item()
                vbatches += 1

        if is_distributed:
            stats = torch.tensor([vloss, float(vbatches)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            vloss, vbatches = stats[0].item(), int(stats[1].item())

        val_loss = vloss / max(vbatches, 1)
        if is_main:
            means = {k: (v[0] / max(len(val_loader), 1), v[1] / max(len(val_loader), 1))
                     for k, v in acc_sum.items()}
            history.log(split="val", epoch=epoch, step=global_step, loss=val_loss,
                        **{f"top1_{k}": v[0] for k, v in means.items()},
                        **{f"top5_{k}": v[1] for k, v in means.items()})
            print(f"epoch {epoch} val   | loss {val_loss:.4f}  "
                  + "  ".join(f"{k} top1 {v[0]:.3f}" for k, v in means.items()))

            _save(os.path.join(out_dir, f"transformer_ep{epoch}.pt"), model, optimizer,
                  scheduler, scaler, cfg, epoch, global_step, is_distributed)
            if val_loss < best_val:
                best_val = val_loss
                _save(os.path.join(out_dir, "transformer_best.pt"), model, optimizer,
                      scheduler, scaler, cfg, epoch, global_step, is_distributed)
                print(f"  best so far (val loss {val_loss:.4f})")

    cleanup_distributed(is_distributed)
    return out_dir
