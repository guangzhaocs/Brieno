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
    "rng_state",
    "restore_rng_state",
    "load_checkpoint",
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


def rng_state():
    """Every random generator a training run draws from, in a form a checkpoint can hold.

    A resumed run must continue the random streams, not restart them: a fresh process
    re-seeds from the run's seed, so without this every resume replays the data order
    from the top of the run, and a run split into short segments sees the same chunks
    again and again.

    Only tensors and plain Python types are stored. ``np.random.get_state()`` contains an
    array, which ``torch.load`` refuses to read back under its ``weights_only`` default
    (torch >= 2.6), so it is converted here rather than discovered on the first resume.
    """
    ns = np.random.get_state()
    version, internal, gauss = random.getstate()
    state = {
        "torch": torch.get_rng_state(),
        "numpy": [ns[0], [int(x) for x in ns[1]], int(ns[2]), int(ns[3]), float(ns[4])],
        "python": [version, list(internal), gauss],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Inverse of :func:`rng_state`. Generators missing from ``state`` are left alone."""
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        ns = state["numpy"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), int(ns[2]),
                             int(ns[3]), float(ns[4])))
    if "python" in state:
        version, internal, gauss = state["python"]
        random.setstate((version, tuple(internal), gauss))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def load_checkpoint(path, map_location="cpu"):
    """``torch.load`` for a checkpoint one of the trainers wrote.

    Tries the safe loader first. A checkpoint written before :func:`rng_state` existed
    may hold a numpy array, which the safe loader rejects; it is the user's own file, so
    it is then read the permissive way instead of being declared unusable.
    """
    import pickle

    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)
