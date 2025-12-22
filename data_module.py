import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
from dataset import collate_fn
import random


class G2pDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 256,
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
            self.val_dataset, self.train_dataset = random_split(
                self.dataset, [valid_set_size, train_set_size], g
            )

            sample_size = max(1, int(train_set_size * self.eval_split))
            sample_indices = random.Random(self.seed + 1).sample(
                range(train_set_size), sample_size
            )
            self.train_sampled_dataset = Subset(self.train_dataset, sample_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_sampled_dataloader(self):
        return DataLoader(
            self.train_sampled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
