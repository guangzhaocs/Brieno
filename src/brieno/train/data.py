"""Datasets for tokenizer pretraining.

One pretraining corpus is a directory of merged ``.npy`` files, one per dataset::

    <pretrain_root>/<dataset name>.npy      float16, shape (N, 1, chunk_size)

produced by ``brieno train prep`` (plus a merge step). Each file is split into a
train and a validation part by a fixed seed, so the same validation chunks are held
out on every run and across machines.

Training draws from several datasets at once with configurable weights, which is how
a corpus that is 12 % human and 5 % maize stays that way no matter how large each
file is.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["MergedSignalDataset", "BlendedDataset", "TokenDataset",
           "BlendedTokenDataset", "SpanMaskCollator"]


class MergedSignalDataset(Dataset):
    """One merged ``.npy`` file, split into train / validation by ``split_seed``.

    The file is memory-mapped and opened lazily in each worker process, so a corpus
    far larger than RAM costs nothing to index.

    ``val_size`` is capped at 10 % of the file, so a small dataset does not lose most
    of itself to validation.
    """

    def __init__(self, npy_file_path, mode="train", val_size=25600, split_seed=42):
        self.npy_file_path = str(npy_file_path)

        head = np.load(self.npy_file_path, mmap_mode="r")
        self.original_total_chunks = head.shape[0]
        del head

        actual_val_size = min(val_size, int(self.original_total_chunks * 0.1))

        rng = np.random.default_rng(split_seed)
        all_indices = np.arange(self.original_total_chunks)
        rng.shuffle(all_indices)

        val_indices = np.sort(all_indices[:actual_val_size])
        train_indices = np.sort(all_indices[actual_val_size:])

        self.indices = train_indices if mode == "train" else val_indices
        self.total_chunks = len(self.indices)
        self.arr = None                      # opened per worker in __getitem__

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        if self.arr is None:
            self.arr = np.load(self.npy_file_path, mmap_mode="r")
        return torch.from_numpy(self.arr[self.indices[idx]].copy()).float()


class BlendedDataset(Dataset):
    """Weighted mixture of several datasets, sampled with replacement.

    An "epoch" is ``epoch_length`` draws rather than a pass over the data: every item
    picks a dataset by ``probabilities`` and then a random chunk inside it. This keeps
    the corpus composition fixed and makes epochs comparable between runs.
    """

    def __init__(self, datasets, probabilities, epoch_length=200000):
        assert len(datasets) == len(probabilities)
        self.datasets = datasets
        self.prob_tensor = torch.tensor(probabilities, dtype=torch.double)
        self.prob_tensor = self.prob_tensor / self.prob_tensor.sum()
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        dataset_idx = torch.multinomial(self.prob_tensor, 1).item()
        chosen = self.datasets[dataset_idx]
        item_idx = torch.randint(0, len(chosen), (1,)).item()
        return chosen[item_idx]


# ----------------------------------------------------------------- token datasets
class TokenDataset(Dataset):
    """One tokenized dataset: ``(N, num_vq_layers, seq_len)`` uint16.

    Split into train / validation exactly like :class:`MergedSignalDataset`, by the
    same fixed seed, so a dataset's held-out part is the same whether you are training
    the tokenizer on its signal or the transformer on its tokens.
    """

    def __init__(self, npy_file_path, mode="train", val_size=25600, split_seed=42):
        self.npy_file_path = str(npy_file_path)

        head = np.load(self.npy_file_path, mmap_mode="r")
        self.original_total_chunks = head.shape[0]
        del head

        actual_val_size = min(val_size, int(self.original_total_chunks * 0.1))

        rng = np.random.default_rng(split_seed)
        all_indices = np.arange(self.original_total_chunks)
        rng.shuffle(all_indices)

        val_indices = np.sort(all_indices[:actual_val_size])
        train_indices = np.sort(all_indices[actual_val_size:])

        self.indices = train_indices if mode == "train" else val_indices
        self.total_chunks = len(self.indices)
        self.arr = None

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        if self.arr is None:
            self.arr = np.load(self.npy_file_path, mmap_mode="r")
        tokens = self.arr[self.indices[idx]].astype(np.int64).copy()
        return torch.from_numpy(tokens)                      # [num_vq_layers, seq_len]


class BlendedTokenDataset(Dataset):
    """Weighted mixture of :class:`TokenDataset`, sized so one epoch is a fixed
    number of optimizer steps per GPU.

    Length is ``epoch_length * batch_size * world_size``: after the distributed
    sampler splits it, every GPU sees exactly ``epoch_length`` batches, which keeps
    "epoch" meaning the same thing however many GPUs are used.
    """

    def __init__(self, datasets, weights, epoch_length, batch_size, world_size):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.prob_tensor = torch.tensor(weights, dtype=torch.float)
        self.epoch_samples = epoch_length * batch_size * world_size

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        ds_idx = torch.multinomial(self.prob_tensor, 1).item()
        sample_idx = torch.randint(0, len(self.datasets[ds_idx]), (1,)).item()
        return self.datasets[ds_idx][sample_idx]             # [num_vq_layers, seq_len]


class SpanMaskCollator:
    """BERT-style span masking over token sequences, applied as a ``collate_fn``.

    The same positions are masked in **every** quantizer layer at once: masking a
    position in one layer only would leave the model the same signal through another
    codebook, and it would learn to copy rather than to predict.

    ``mask_start_prob`` corrects for span overlap, so the realised fraction of masked
    positions matches ``mask_ratio`` rather than overshooting it by ``min_span``.
    Replacement follows BERT's 80 / 10 / 10 split: mask token, random token, unchanged.
    """

    def __init__(self, mask_ratio, min_span, mask_token_id, vocab_size):
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_start_prob = 1.0 - (1.0 - mask_ratio) ** (1.0 / min_span)

    def __call__(self, batch):
        tokens = torch.stack(batch).long()                   # [B, Q, L]
        B, Q, L = tokens.shape
        input_ids = tokens.clone()
        labels = torch.full((B, Q, L), -100, dtype=torch.long)

        starts = torch.rand(B, L) < self.mask_start_prob
        is_masked = starts.clone()
        for k in range(1, self.min_span):                    # grow starts into spans
            is_masked[:, k:] = is_masked[:, k:] | starts[:, :-k]

        mask_expanded = is_masked.unsqueeze(1).expand(B, Q, L)
        labels[mask_expanded] = tokens[mask_expanded]

        rand_vals = torch.rand(B, L)
        replace_mask = is_masked & (rand_vals < 0.8)
        random_mask = is_masked & (rand_vals >= 0.8) & (rand_vals < 0.9)

        input_ids[replace_mask.unsqueeze(1).expand(B, Q, L)] = self.mask_token_id
        if random_mask.any():
            rand_tokens = torch.randint(0, self.vocab_size, (B, Q, L))
            sel = random_mask.unsqueeze(1).expand(B, Q, L)
            input_ids[sel] = rand_tokens[sel]
        # the remaining 10 % keep their original token, already copied above
        return input_ids, labels
