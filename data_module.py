import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler, random_split, Subset
import torch
from dataset import collate_fn
import random


class BucketBatchSampler(Sampler):
    """Groups similar-length sequences into batches while shuffling each epoch."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        # Add slight length jitter so batch compositions vary across epochs
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i] + rng.uniform(-1.5, 1.5))

        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class G2pDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 128,
        eval_split: float = 0.05,
        seed: int = 42,
        num_workers: int = 8,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.eval_split = eval_split
        self.seed = seed
        self.num_workers = num_workers

        self.train_dataset = None
        self.train_lengths = None
        self.train_sampled_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            valid_set_size = int(len(self.dataset) * self.eval_split)
            train_set_size = len(self.dataset) - valid_set_size
            assert valid_set_size > 0
            assert train_set_size > 0

            g = torch.Generator()
            g.manual_seed(self.seed)
            raw_val_dataset, self.train_dataset = random_split(
                self.dataset, [valid_set_size, train_set_size], g
            )

            # Sort validation set by length so greedy_decode exits early on short words
            val_sorted_indices = sorted(
                range(valid_set_size),
                key=lambda i: len(self.dataset.entries[raw_val_dataset.indices[i]][0]),
            )
            self.val_dataset = Subset(raw_val_dataset, val_sorted_indices)

            self.train_lengths = [
                len(self.dataset.entries[idx][0]) for idx in self.train_dataset.indices
            ]

            sample_size = max(1, int(train_set_size * self.eval_split))
            sample_indices = sorted(
                random.Random(self.seed + 1).sample(range(train_set_size), sample_size),
                key=lambda i: self.train_lengths[i],
            )
            self.train_sampled_dataset = Subset(self.train_dataset, sample_indices)

    def train_dataloader(self):
        sampler = BucketBatchSampler(
            self.train_lengths, batch_size=self.batch_size, seed=self.seed
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_sampled_dataloader(self):
        return DataLoader(
            self.train_sampled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )