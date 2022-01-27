from unicodedata import name
import torch
import torch.nn as nn
import torchvision  # TODO: Use some pretrained model for transfer learning.
from typing import List, Tuple
from torch.utils.data import DataLoader
from pathlib import Path


class LinBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm_layer: nn.Module = nn.BatchNorm1d,
        activation: nn.Module = nn.ReLU,
        dropout: int = 0,
    ) -> None:
        """Linear block.

        Args:
            in_features (int): Input dimensionality.
            out_features (int): Output dimensionality.
            norm_layer (nn.Module, optional): Normalization layer function. Defaults to nn.BatchNorm1d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super().__init__()

        self.net = nn.Sequential()

        if dropout:
            self.net.add_module(name="dropout", module=nn.Dropout(p=dropout))

        self.net.add_module(
            name="linear",
            module=nn.Linear(
                in_features=in_features, out_features=out_features, bias=True
            ),
        )

        if norm_layer:
            self.net.add_module(name="norm", module=norm_layer)

        if activation:
            self.net.add_module(name="activation", module=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        features: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
        dropouts: List[int],
    ) -> None:
        """Multi layer perceptron.

        Args:
            in_size (int): Input size.
            features (List[int]): List of linear layers' features.
            norm_layers (List[nn.Module]): List of normalization layers.
            activations (List[nn.Module]): List of activations functions.
            dropouts (List[int]): List of dropout probabilities.
        """
        super().__init__()

        features = [in_size] + features

        self.net = nn.Sequential()

        for n, (i, o, norm, act, p) in enumerate(
            zip(features[:-1], features[1:], norm_layers, activations, dropouts)
        ):
            self.net.add_module(
                "lin" + str(n),
                LinBlock(
                    in_features=i,
                    out_features=o,
                    norm_layer=norm,
                    activation=act,
                    dropout=p,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)
