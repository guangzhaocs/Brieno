"""Encode signal chunks into discrete tokens with a trained tokenizer.

The transformer pretrains on token sequences, not on signal, so a corpus has to be
run through the tokenizer once::

    <dataset>.npy   (N, 1, chunk_size)   float16 signal
        -> encode ->
    <dataset>.npy   (N, num_quantizers, seq_len)   uint16 token ids

Written incrementally through a memory-mapped output, so peak memory stays flat
whatever the size of the corpus.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["load_tokenizer", "encode_dataset"]


class _SignalChunks(Dataset):
    """Read-only view of a merged signal ``.npy``, opened lazily per worker."""

    def __init__(self, path):
        self.path = str(path)
        head = np.load(self.path, mmap_mode="r")
        self.n = head.shape[0]
        del head
        self.arr = None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(self.arr[idx].copy()).float()


def load_tokenizer(checkpoint_path, device="cuda"):
    """Rebuild a :class:`SignalTokenizer` from a checkpoint written by ``brieno train tokenizer``.

    The checkpoint carries its own ``model_config``, so no separate config file is
    needed and the architecture can never silently disagree with the weights.
    """
    from brieno.models.tokenizer import SignalTokenizer

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in state or "model_state_dict" not in state:
        raise ValueError(
            f"{checkpoint_path} is not a brieno tokenizer checkpoint "
            "(expected 'model_config' and 'model_state_dict')")
    sd = state["model_state_dict"]
    model = SignalTokenizer.from_config(state["model_config"], kmeans_init=False)

    # `init_buffer` collects encoder outputs until there are enough for the codebook's
    # k-means initialisation, then is emptied. Its length therefore differs between a
    # fresh model and a checkpoint taken mid-collection, so match the saved shape
    # before loading rather than failing on it.
    for name, tensor in sd.items():
        if name.endswith("init_buffer"):
            *path, attr = name.split(".")
            module = model
            for part in path:
                module = getattr(module, part)
            setattr(module, attr, torch.empty_like(tensor))
    model.load_state_dict(sd)

    # A codebook whose k-means never ran is still random: encoding with it would
    # silently produce meaningless tokens, which is worse than refusing.
    uninitialised = [name for name, tensor in sd.items()
                     if name.endswith(".inited") and not bool(tensor.reshape(-1)[0])]
    if uninitialised:
        raise ValueError(
            f"{checkpoint_path}: the codebook was never initialised "
            f"(k-means had not completed by epoch {state.get('epoch', '?')}; "
            f"{len(uninitialised)} of {state['model_config']['num_quantizers']} quantizers). "
            "Encode with a fully trained tokenizer.")

    return model.to(device).eval(), state["model_config"]


def encode_dataset(checkpoint_path, input_path, output_path, *, batch_size=512,
                   num_workers=4, device="cuda", force=False, progress=True):
    """Tokenize one merged signal file. Returns ``(n_chunks, num_quantizers, seq_len)``."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; use --force to overwrite")

    model, model_cfg = load_tokenizer(checkpoint_path, device=device)
    n_quant = int(model_cfg["num_quantizers"])

    data = _SignalChunks(input_path)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

    # One forward pass fixes seq_len, which depends on the tokenizer's downsample.
    with torch.no_grad():
        probe = next(iter(loader)).to(device)
        _, tokens, _, _ = model(probe)
        seq_len = int(tokens.shape[2])            # tokens: [num_quantizers, B, seq_len]

    print(f" * {len(data):,} chunks -> ({len(data)}, {n_quant}, {seq_len}) uint16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16,
                                    shape=(len(data), n_quant, seq_len))

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"  {output_path.name}", leave=False)
        except ImportError:
            pass

    at = 0
    with torch.no_grad():
        for waveforms in iterator:
            waveforms = waveforms.to(device, non_blocking=True)
            _, tokens, _, _ = model(waveforms)                 # [Q, B, L]
            batch = tokens.permute(1, 0, 2).to(torch.int32).cpu().numpy()
            out[at:at + batch.shape[0]] = batch.astype(np.uint16)
            at += batch.shape[0]
    out.flush()
    del out

    assert at == len(data), f"wrote {at} chunks, expected {len(data)}"
    print(f" * wrote {output_path}")
    return len(data), n_quant, seq_len
