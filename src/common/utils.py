import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# * Unused but could still be useful.
class OneHot(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """One-hot encoder as a layer.

        Args:
            num_classes (int): Number of classes.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): A tensor of class labels.

        Returns:
            torch.Tensor: The input tensor encoded as a batch of one-hot vectors.
        """
        return F.one_hot(x, num_classes=self.num_classes)


if __name__ == "__main__":
    pass
