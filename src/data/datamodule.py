from torch.utils.data import DataLoader
from .dataset import EmotionDataset
from .transforms import get_train_transforms, get_eval_transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

class EmotionDataModule:
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        img_size=224,
        use_weighted_sampler=False
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_weighted_sampler = use_weighted_sampler

    def setup(self):
        self.train_dataset = EmotionDataset(
            root=f"{self.data_dir}/train",
            transform=get_train_transforms(self.img_size)
        )
        self.val_dataset = EmotionDataset(
            root=f"{self.data_dir}/val",
            transform=get_eval_transforms(self.img_size)
        )
        self.test_dataset = EmotionDataset(
            root=f"{self.data_dir}/test",
            transform=get_eval_transforms(self.img_size)
        )

        self.train_sampler = None

        if self.use_weighted_sampler:
            targets = self.train_dataset.targets
            class_counts = Counter(targets)

            class_weights = {
                cls: 1.0 / count
                for cls, count in class_counts.items()
            }

            sample_weights = torch.tensor(
                [class_weights[t] for t in targets],
                dtype=torch.float
            )

            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler = self.train_sampler,
            shuffle=not self.use_weighted_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
