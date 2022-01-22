from pyexpat import features, model
from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torchvision  # TODO: Use some pretrained model for transfer learning.
from typing import List, Tuple
from src.common.utils import get_dataset
from torch.utils.data import DataLoader
from pathlib import Path


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        """Define a simple block that does convolution, normalization, etc...

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output feature maps.
            kernel_size (Tuple[int, int]): Size of the kernel.
            stride (int): Stride value.
            padding (int): Padding size.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = norm_layer(num_features=out_channels)

        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.cnn(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        """Define a simple block that does transposed convolution, normalization, etc...

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output feature maps.
            kernel_size (Tuple[int, int]): Size of the kernel.
            stride (int): Stride value.
            padding (int): Padding size.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.cnn = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = norm_layer(num_features=out_channels)

        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.cnn(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
    ) -> None:
        """Convolutional neural network block.

        Args:
            in_channels (int): Number of channels in input to the first layer.
            features (List[int]): A list containing the output feature maps of each convolutional layer.
            kernels (List[int]): A list containing the kernel sizes in each convolutional layer.
            strides (List[int]): A list of strides in each convolutional layer.
            paddings (List[int]): A list of paddings in each convolutional layer.
            norm_layers (nn.Module, optional): List of normalization layers.
            activations (nn.Module, optional): List of activation functions.
        """
        super().__init__()

        # Doing that we can manage multuple conv layers more easily.
        channels = [in_channels] + features

        self.net = nn.Sequential()
        for n, (i, o, k, s, p, norm, act) in enumerate(
            zip(
                channels[:-1],
                channels[1:],
                kernels,
                strides,
                paddings,
                norm_layers,
                activations,
            )
        ):
            self.net.add_module(
                "conv" + str(n),
                ConvBlock(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=[k, k],
                    stride=s,
                    padding=[p, p],
                    norm_layer=norm,
                    activation=act,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.net(x)
        return out


class ConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int],
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
    ) -> None:
        """Transposed convolutional neural network block.

        Args:
            in_channels (int): Number of channels in input to the first layer.
            features (List[int]): A list containing the output feature maps of each convolutional layer.
            kernels (List[int]): A list containing the kernel sizes in each convolutional layer.
            strides (List[int]): A list of strides in each convolutional layer.
            paddings (List[int]): A list of paddings in each convolutional layer.
            norm_layers (nn.Module, optional): List of normalization layers.
            activations (nn.Module, optional): List of activation functions.

        """
        super().__init__()

        # Doing that we can manage multuple conv layers more easily.
        channels = [in_channels] + features

        self.net = nn.Sequential()
        for n, (i, o, k, s, p, norm, act) in enumerate(
            zip(
                channels[:-1],
                channels[1:],
                kernels,
                strides,
                paddings,
                norm_layers,
                activations,
            )
        ):
            self.net.add_module(
                "conv" + str(n),
                ConvTransposeBlock(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=[k, k],
                    stride=s,
                    padding=[p, p],
                    norm_layer=norm,
                    activation=act,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.net(x)
        return out


if __name__ == "__main__":
    ROOT = Path(".")
    ds = get_dataset(root=ROOT / "data", name="mnist", download=False)
    data_loader = DataLoader(dataset=ds, batch_size=16, shuffle=True, drop_last=False)

    conv = Conv(
        in_channels=1,
        features=[5, 5, 5, 5, 5, 1],
        kernels=[3, 5, 1, 3, 5, 3],
        strides=[1, 2, 1, 1, 2, 1],
        paddings=[0] * 6,
        norm_layers=[nn.BatchNorm2d] * 6,
        activations=[nn.ReLU] * 6,
    )

    conv_t = ConvTranspose(
        in_channels=3,
        features=[5, 3, 5, 3, 1],
        kernels=[3, 5, 5, 3, 2],
        strides=[2, 2, 1, 1, 1],
        paddings=[0, 0, 0, 0, 0],
        norm_layers=[nn.BatchNorm2d] * 5,
        activations=[nn.ReLU] * 5,
    )

    for xb in data_loader:
        x = xb[0]
        y = xb[1]
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")

        out = conv(x)
        print(f"conv: {out.shape}")
        # out = model(x)
        # print(f"model: {out.shape}")

        # z = torch.rand(16, 48).reshape(16, 3, 4, 4)
        # out = conv_t(z)
        # print(f"conv_t: {out.shape}")
        break
