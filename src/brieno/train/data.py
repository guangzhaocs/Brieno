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

__all__ = ["MergedSignalDataset", "BlendedDataset"]


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
