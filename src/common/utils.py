from ast import Raise
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


def get_torch_dataset(opt, transform=[]) -> torchvision.datasets:
    """Wrapper that for a dataset's name return the correspind torchvision loader.

    Args:
        name (str): Dataset's name.

    Returns:
        torchvision.datasets: Torchvision's loading function.
    """
    # By default z-score normalization does not change anything.
    mean = [0]
    std = [1]

    # ! There is an issue for which it cannot be downloaded but should be fixed in the next PyTorch stable release.
    if opt.dataset == "celeb_a":
        dataset = torchvision.datasets.CelebA(
            root=opt.data_dir,
            split="train",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    if opt.dataset == "cifar10":
        if opt.normalization:
            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]
        dataset = datasets.CIFAR10(
            root=opt.data_dir,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=opt.data_dir,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    elif opt.dataset == "fashion-mnist":
        if opt.normalization:
            mean = [0.2860402]
            std = [0.3530239]
        dataset = torchvision.datasets.FashionMNIST(
            root=opt.data_dir,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    elif opt.dataset == "imagenet":
        dataset = torchvision.datasets.ImageNet(
            root=opt.data_dir, split="train", download=opt.download
        )
    elif opt.dataset == "mnist":
        if opt.normalization:
            mean = [0.1307]
            std = [0.3081]
        dataset = torchvision.datasets.MNIST(
            root=opt.data_dir,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    elif opt.dataset == "omniglot":
        dataset = torchvision.datasets.Omniglot(
            root=opt.data_dir,
            background=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
                + transform
            ),
            download=opt.download,
        )
    else:
        raise Exception("Non valid dataset.")

    # We always convert to tensor and normalize.
    return dataset
