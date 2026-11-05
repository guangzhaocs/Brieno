"""Tokenizer pretraining: reconstruct raw signal through a residual vector quantizer.

The objective is ``L1(reconstruction, signal) + vq_loss``; there is no label of any
kind, so any corpus of raw reads will do. The codebooks are EMA-tracked, which is why
the optimizer only sees the encoder and decoder.

Run on one GPU::

    brieno train tokenizer -c config.json

or on several with torchrun::

    torchrun --nproc_per_node=4 -m brieno.cli train tokenizer -c config.json
"""
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from brieno.models.tokenizer import SignalTokenizer
from brieno.train.data import BlendedDataset, MergedSignalDataset
from brieno.train.utils import (
    JsonlLogger,
    autocast,
    cleanup_distributed,
    cosine_schedule_with_warmup,
    grad_scaler,
    setup_distributed,
    worker_init_fn,
)

__all__ = ["build_train_dataset", "build_val_dataset", "pretrain_tokenizer"]


def build_train_dataset(cfg, verbose=True):
    """Weighted blend of the training splits named in ``datasets.pretrain_weights``."""
    root = cfg["datasets"]["pretrain_root"]
    val_size = cfg["datasets"]["validation_size_per_dataset"]

    datasets, weights = [], []
    for entry in cfg["datasets"]["pretrain_weights"]:
        path = os.path.join(root, entry["name"] + ".npy")
        ds = MergedSignalDataset(path, mode="train", val_size=val_size)
        datasets.append(ds)
        weights.append(float(entry["weight"]))
        if verbose:
            print(f"  [train] {entry['name']:50s} chunks={len(ds):>10,d}  weight={entry['weight']}")

    total = sum(weights)
    weights = [w / total for w in weights]
    return BlendedDataset(datasets, weights, epoch_length=cfg["training"]["epoch_length"])


def build_val_dataset(cfg, verbose=True):
    """Concatenated validation splits of the datasets named in ``datasets.validation``.

    These are the held-out chunks of the same files, so validation covers datasets that
    training never samples from as well.
    """
    root = cfg["datasets"]["pretrain_root"]
    val_size = cfg["datasets"]["validation_size_per_dataset"]

    parts = []
    for name in cfg["datasets"]["validation"]:
        ds = MergedSignalDataset(os.path.join(root, name + ".npy"), mode="val", val_size=val_size)
        parts.append(ds)
        if verbose:
            print(f"  [val]   {name:50s} chunks={len(ds):>10,d}")
    return ConcatDataset(parts)


def _save(path, model, optimizer, scheduler, scaler, cfg, epoch, global_step, is_distributed):
    """Checkpoint holding a plain tokenizer state_dict plus everything needed to resume."""
    torch.save({
        "format": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": cfg["model"],
        "model_state_dict": (model.module if is_distributed else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)


def pretrain_tokenizer(config_path, seed=42, run_name=None, resume=None, overrides=None):
    """Pretrain a :class:`SignalTokenizer` from a JSON config. Returns the output directory."""
    is_distributed, local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = global_rank == 0

    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for section, key, value in (overrides or []):
        if value is not None:
            cfg[section][key] = value

    # One seed per rank: with the same seed every rank would draw the same chunks.
    torch.manual_seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    random.seed(seed + global_rank)

    # ---------------------------------------------------------------- output directory
    if is_main:
        name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(cfg["training"]["save_dir"], name)
        os.makedirs(out_dir, exist_ok=True)
        cfg["training"]["save_dir"] = out_dir
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"writing to {out_dir}")

    if is_distributed:                       # rank 0 chose the directory; everyone needs it
        dist.barrier()
        box = [cfg]
        dist.broadcast_object_list(box, src=0)
        cfg = box[0]
        dist.barrier()

    out_dir = cfg["training"]["save_dir"]
    history = JsonlLogger(os.path.join(out_dir, "history.jsonl"), enabled=is_main)

    # ---------------------------------------------------------------- data
    if is_main:
        print("\nbuilding datasets ...")
    train_set = build_train_dataset(cfg, verbose=is_main)
    val_set = build_val_dataset(cfg, verbose=is_main)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank,
                                       shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=global_rank,
                                     shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=cfg["training"]["num_workers"], pin_memory=True,
                              drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
                            sampler=val_sampler, num_workers=cfg["training"]["num_workers"],
                            pin_memory=True, drop_last=False)

    # ---------------------------------------------------------------- model
    model = SignalTokenizer.from_config(cfg["model"], kmeans_init=True).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\ntokenizer: downsample {model.downsample}x -> {16384 // model.downsample} frames"
              f" | {model.quantizer_type} {model.num_quantizers}x{cfg['model']['num_tokens']}"
              + (f" in a {model.quant_dim}-d projected space" if model.has_projection else "")
              + f" | {n_params / 1e6:.1f} M parameters")

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    total_steps = epochs * len(train_loader)
    warmup = cfg["training"].get("warmup_steps", int(total_steps * 0.05))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = grad_scaler()
    recon_criterion = nn.L1Loss()

    start_epoch, global_step, best_val = 1, 0, float("inf")
    if resume:
        state = torch.load(resume, map_location=device)
        (model.module if is_distributed else model).load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        if is_main:
            print(f"resumed from {resume} at epoch {state['epoch']}")

    n_tokens = int(cfg["model"]["num_tokens"])
    n_quant = int(cfg["model"]["num_quantizers"])
    commit = float(cfg["model"]["commitment_cost"])
    log_every = int(cfg.get("logging", {}).get("log_every_n_steps", 50))

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "vq": 0.0}
        t0 = time.time()

        for step, waveforms in enumerate(train_loader):
            waveforms = waveforms.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                reconstructed, tokens, vq_loss, vq_per_layer = model(waveforms)
                recon_loss = recon_criterion(reconstructed, waveforms)
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # A step the scaler skipped (inf/NaN gradients) must not advance the LR.
            if scale_before <= scaler.get_scale():
                scheduler.step()

            logs = [loss.detach(), recon_loss.detach(), vq_loss.detach()]
            if is_distributed:
                for t in logs:
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= world_size
            sums["loss"] += logs[0].item()
            sums["recon"] += logs[1].item()
            sums["vq"] += logs[2].item()
            global_step += 1

            if is_main and step % log_every == 0:
                active = [int(torch.unique(tokens[i]).numel()) for i in range(tokens.shape[0])]
                history.log(split="train", epoch=epoch, step=global_step,
                            loss=logs[0].item(), recon=logs[1].item(), vq=logs[2].item(),
                            vq_mse=logs[2].item() / commit,
                            lr=optimizer.param_groups[0]["lr"],
                            active_tokens=active,
                            dead_rate=[(n_tokens - a) / n_tokens for a in active],
                            vq_per_layer=[v.item() for v in vq_per_layer])
                print(f"  epoch {epoch}/{epochs}  step {step}/{len(train_loader)}  "
                      f"loss {logs[0].item():.4f}  recon {logs[1].item():.4f}  "
                      f"vq {logs[2].item():.4f}  lr {optimizer.param_groups[0]['lr']:.2e}",
                      flush=True)

        n_steps = max(len(train_loader), 1)
        if is_main:
            print(f"epoch {epoch} train | loss {sums['loss']/n_steps:.4f}  "
                  f"recon {sums['recon']/n_steps:.4f}  vq {sums['vq']/n_steps:.4f}  "
                  f"({time.time()-t0:.0f}s)")

        # ------------------------------------------------------------ validation
        model.eval()
        vsums = {"loss": 0.0, "recon": 0.0, "vq": 0.0}
        token_counts = torch.zeros((n_quant, n_tokens), device=device)

        with torch.no_grad():
            for waveforms in val_loader:
                waveforms = waveforms.to(device, non_blocking=True)
                with autocast():
                    reconstructed, tokens, vq_loss, _ = model(waveforms)
                    recon_loss = recon_criterion(reconstructed, waveforms)
                    loss = recon_loss + vq_loss

                for layer in range(tokens.shape[0]):
                    token_counts[layer] += torch.bincount(tokens[layer].reshape(-1),
                                                          minlength=n_tokens).float()

                logs = [loss.clone(), recon_loss.clone(), vq_loss.clone()]
                if is_distributed:
                    for t in logs:
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        t /= world_size
                vsums["loss"] += logs[0].item()
                vsums["recon"] += logs[1].item()
                vsums["vq"] += logs[2].item()

        if is_distributed:
            dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)

        n_val = max(len(val_loader), 1)
        val_loss = vsums["loss"] / n_val

        if is_main:
            probs = token_counts / (token_counts.sum(dim=1, keepdim=True) + 1e-10)
            perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum(dim=1))
            active = (token_counts > 0).sum(dim=1)
            history.log(split="val", epoch=epoch, step=global_step, loss=val_loss,
                        recon=vsums["recon"] / n_val, vq=vsums["vq"] / n_val,
                        vq_mse=(vsums["vq"] / n_val) / commit,
                        perplexity=[float(p) for p in perplexity],
                        active_tokens=[int(a) for a in active],
                        dead_rate=[float((n_tokens - a) / n_tokens) for a in active])
            print(f"epoch {epoch} val   | loss {val_loss:.4f}  "
                  f"recon {vsums['recon']/n_val:.4f}  vq {vsums['vq']/n_val:.4f}  "
                  f"active {[int(a) for a in active]}  "
                  f"perplexity {[round(float(p), 1) for p in perplexity]}")

            _save(os.path.join(out_dir, f"tokenizer_ep{epoch}.pt"), model, optimizer,
                  scheduler, scaler, cfg, epoch, global_step, is_distributed)
            if val_loss < best_val:
                best_val = val_loss
                _save(os.path.join(out_dir, "tokenizer_best.pt"), model, optimizer,
                      scheduler, scaler, cfg, epoch, global_step, is_distributed)
                print(f"  best so far (val loss {val_loss:.4f})")

        if epoch == cfg["training"].get("early_stop", epochs):
            if is_main:
                print(f"stopping at epoch {epoch}")
            break

    cleanup_distributed(is_distributed)
    return out_dir
