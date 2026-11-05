"""Shared training helpers: distributed setup, LR schedule, worker seeding, logging."""
import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "worker_init_fn",
    "cosine_schedule_with_warmup",
    "autocast",
    "grad_scaler",
    "JsonlLogger",
]


def setup_distributed():
    """Join the torchrun process group when there is one.

    Returns ``(is_distributed, local_rank, global_rank, world_size)``. Without
    ``LOCAL_RANK`` in the environment this is a single-process run on GPU 0 (or CPU).
    """
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, global_rank, world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return False, 0, 0, 1


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def worker_init_fn(worker_id):
    """Give every dataloader worker its own numpy / random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Linear warmup, then cosine decay to zero."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def autocast():
    """fp16 autocast on CUDA, under whichever torch API this version provides."""
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):           # torch < 2.0
        return torch.cuda.amp.autocast()


def grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):           # torch < 2.3
        return torch.cuda.amp.GradScaler()


class JsonlLogger:
    """Append-only metric log: one JSON object per line.

    Deliberately not a third-party experiment tracker — a run's history stays in the
    output directory, readable without an account or a network connection.
    """

    def __init__(self, path, enabled=True):
        self.path = str(path)
        self.enabled = enabled
        if enabled:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def log(self, **record):
        if not self.enabled:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
