import numpy as np
import torchvision
from typing import Union, Tuple
from pathlib import Path


def get_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get mean and standard deviation from a numpy dataset.
    :param data: Input data.
    :return: Mean and std.
    """
    return np.mean(data), np.std(data)


def get_dataset(
    name: str, root: Union[str, Path], download: bool
) -> torchvision.datasets:
    """
    Wrapper to get the correct torchvision dataset function.
    :param name: Name of the dataset we want to work with.
    :param root: Parent directory in which we should look for the dataset.
    :param download: If true, download data from the internet.
    :return: The torchvision function.
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

    return dataset_fn(
        root=root,
        train=True,  # We always consider the train set only.
        download=download,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        ),
    )


if __name__ == "__main__":
    pass
