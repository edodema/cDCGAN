import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from typing import Union, Tuple
from pathlib import Path


def get_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get mean and standard deviation from a numpy dataset.
    :param data: Input data.
    :return: Mean and std.
    """
    return np.mean(data), np.std(data)


def get_torch_dataset(opt) -> torchvision.datasets:
    """Wrapper that for a dataset's name return the correspind torchvision loader.

    Args:
        name (str): Dataset's name.

    Returns:
        torchvision.datasets: Torchvision's loading function.
    """
    # By default z-score normalization does not change anything.
    mean = [0]
    std = [1]

    if opt.dataset == "cifar10":
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        dataset_fn = datasets.CIFAR10(
            root=opt.data_dir,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            ),
        )
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
