from logging import root
from pathlib import Path
from typing import *
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TorchDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict):
        """Datamodule for some datasets downloadble from torchvision.

        Args:
            cfg (Dict): Configuration object.
        """
        super().__init__()

        self.fn, transforms = TorchDataModule.get_torch_dataset(name=cfg["name"])

        self.root = cfg["root"]
        self.transforms = torchvision.transforms.Compose(transforms + cfg["transforms"])

        self.num_workers = cfg["num_workers"]
        self.batch_size = cfg["batch_size"]
        self.shuffle = cfg["shuffle"]

    def prepare_data(self) -> None:
        """Downlaod datasets."""
        self.fn(root=self.root, train=True, download=True)
        self.fn(root=self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets.

        Args:
            stage (Optional[str], optional): Defines if we are training/validating or testing. Defaults to None.
        """
        if stage == "fit" or stage is None:
            ds = self.fn(root=self.root, train=True, transform=self.transforms)
            self.ds_train, self.ds_val = random_split(
                dataset=ds,
                lengths=[int(0.8 * len(ds)), int(0.2 * len(ds))],
            )

        if stage == "test" or stage is None:
            self.ds_test = self.fn(
                root=self.root, train=False, transform=self.transforms
            )

    def train_dataloader(self) -> DataLoader:
        """Get train's dataloader.

        Returns:
            DataLoader: Dataloader object.
        """
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation's dataloader.

        Returns:
            DataLoader: Dataloader object.
        """
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test's dataloader.

        Returns:
            DataLoader: Dataloader object.
        """
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    @staticmethod
    def get_torch_dataset(name: str) -> torchvision.datasets:
        """Wrapper that for a dataset's name return the correspind torchvision loader.

        Args:
            name (str): Dataset's name.

        Returns:
            torchvision.datasets: Torchvision's loading function.
        """
        dataset_fn = None

        # By default z-score normalization does not change anything.
        mean = [0]
        std = [1]

        if name == "cifar10":
            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]
            dataset_fn = torchvision.datasets.CIFAR10
        elif name == "cifar100":
            dataset_fn = torchvision.datasets.CIFAR100
        elif name == "fashion-mnist":
            dataset_fn = torchvision.datasets.FashionMNIST
        elif name == "imagenet":
            dataset_fn = torchvision.datasets.ImageNet
        elif name == "mnist":
            mean = [0.1307]
            std = [0.3081]
            dataset_fn = torchvision.datasets.MNIST
        elif name == "omniglot":
            dataset_fn = torchvision.datasets.Omniglot

        # We always convert to tensor and normalize.
        return dataset_fn, [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
