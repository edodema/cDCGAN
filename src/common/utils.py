import numpy as np
import torch
import torchvision
from typing import List, Optional, Union
from pathlib import Path


def get_stats(data: np.ndarray):
    return np.mean(data), np.std(data)


def get_dataset(name: str, root: Union[str, Path], download: bool):
    dataset_fn = None
    mean = [0, 0, 0]
    std = [1, 1, 1]

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
        dataset_fn = torchvision.datasets.MNIST
    elif name == "omniglot":
        dataset_fn = torchvision.datasets.Omniglot

    return dataset_fn(
        root=root,
        train=True,
        download=download,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        ),
    )


if __name__ == "__main__":
    root = Path("../../data")
    print(root.exists())
