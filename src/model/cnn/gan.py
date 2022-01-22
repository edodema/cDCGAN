import torch
import torch.nn as nn
from typing import List
from src.model.cnn.cnn import Conv, ConvTranspose
from src.common.utils import get_dataset, OneHot
from torch.utils.data import DataLoader
from pathlib import Path


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conditional_size: int,
        features: List[int],
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
    ) -> None:
        """Discriminator.

        Args:
            in_channels (int): Number of channels in input images.
            conditional_size (int): Size of the conditional vector, we assume a one-hot encoding therefore this number will match the number of classes.
            features (List[int]): A list containing the output feature maps of each convolutional layer.
            kernels (List[int]): A list containing the kernel sizes in each convolutional layer.
            strides (List[int]): A list of strides in each convolutional layer.
            paddings (List[int]): A list of paddings in each convolutional layer.
            norm_layers (nn.Module, optional): List of normalization layers.
            activations (nn.Module, optional): List of activation functions.
        """
        super().__init__()

        # Feature extraction backbone.
        self.cnn = Conv(
            in_channels=in_channels,
            features=features,
            kernels=kernels,
            strides=strides,
            paddings=paddings,
            norm_layers=norm_layers,
            activations=activations,
        )

        # One-hot encoder.
        self.oh = OneHot(conditional_size)

        # Dense layer.
        self.fc = nn.Linear(in_features=features[-1] + conditional_size, out_features=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Batch of input images.
            c (torch.Tensor): Batch of conditioning vectors.

        Returns:
            torch.Tensor: Output tensor.
        """
        oh = self.oh(c)
        out = self.cnn(x)

        assert (
            oh.shape == out.shape[:2]
        ), "The CNN should output a number of features equal to `conditional_size`."
        assert out.shape[2:] == torch.Size([1, 1]), "The CNN should reduce H, W to 1."

        out = out.view(out.shape[0], out.shape[1])

        # Combine image and conditioning data.
        out = torch.cat((out, oh), dim=-1)

        # Classification head outputs a probability distribution.
        logits = self.fc(out)
        return logits


class Generator(nn.Module):
    def __init__(
        self,
        in_size: int,
        h_size: int,
        conditional_size: int,
        in_channels: int,
        features: List[int],
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
    ) -> None:
        """Generator.

        Args:
            in_size (int): Side of the input image, we assume it to be a square.
            h_size (int): Size of the hidden representation of input || conditioning.
            conditional_size (int): Size of the conditional vector, we assume a one-hot encoding therefore this number will match the number of classes.
            in_channels (int): Number of channels in input images.
            features (List[int]): A list containing the output feature maps of each convolutional layer.
            kernels (List[int]): A list containing the kernel sizes in each convolutional layer.
            strides (List[int]): A list of strides in each convolutional layer.
            paddings (List[int]): A list of paddings in each convolutional layer.
            norm_layers (nn.Module, optional): List of normalization layers.
            activations (nn.Module, optional): List of activation functions.
        """
        super().__init__()

        self.c = in_channels
        # Used h and w for modularity, for now they are always the same. We can easily support non-square images using lists.
        self.h = h_size
        self.w = h_size

        # Dense layer, could have just reshaped.
        self.fc = nn.Linear(
            in_features=in_size + conditional_size,
            out_features=in_channels * h_size * h_size,
        )

        # One-hot encoder.
        self.oh = OneHot(conditional_size)

        # Image generation bone.
        self.cnn = ConvTranspose(
            in_channels=in_channels,
            features=features,
            kernels=kernels,
            strides=strides,
            paddings=paddings,
            norm_layers=norm_layers,
            activations=activations,
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z (torch.Tensor): Batch of random noise.
            c (torch.Tensor): Batch of conditioning vectors.


        Returns:
            torch.Tensor: Generated image.
        """
        oh = self.oh(c)
        x = torch.cat((z, oh), dim=-1)

        out = self.fc(x)
        out = out.view(out.shape[0], self.c, self.h, self.w)

        out = self.cnn(out)
        return out


if __name__ == "__main__":
    ROOT = Path(".")
    ds = get_dataset(root=ROOT / "data", name="mnist", download=False)
    data_loader = DataLoader(dataset=ds, batch_size=16, shuffle=True, drop_last=False)

    d = Discriminator(
        in_channels=1,
        conditional_size=10,
        features=[32, 16, 16, 12, 11, 10],
        kernels=[3, 5, 1, 3, 5, 3],
        strides=[1, 2, 1, 1, 2, 1],
        paddings=[0] * 6,
        norm_layers=[nn.BatchNorm2d] * 6,
        activations=[nn.ReLU] * 6,
    )

    g = Generator(
        in_size=100,
        h_size=4,
        conditional_size=10,
        in_channels=1,
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
        # print(f"x: {x.shape}")
        # print(f"y: {y.shape}")

        out = d(x, y)
        print(f"d: {out.shape}")

        out = g(z=torch.rand(x.shape[0], 100), c=y)
        print(f"g: {out.shape}")
        break
