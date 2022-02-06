from typing import *
import torch
from torch import nn
import torch.nn.functional as F


class ConvT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        normalization: nn.Module,
        activation: nn.Module,
    ):
        super(ConvT, self).__init__()

        self.convT = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[kernel_size, kernel_size],
            stride=stride,
            padding=padding,
        )
        # In GAN's seminal paper is advised to use a normal distribution.
        torch.nn.init.normal_(self.convT.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(self.convT.bias, val=0.0)

        if normalization:
            self.norm = normalization(out_channels)

        self.act = activation

    def forward(self, x: torch.Tensor):
        out = self.convT(x)

        if hasattr(self, "norm"):
            out = self.norm(out)

        out = self.act(out)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        conditional_dim: int,
        filters: List[int],
        output_dim: int,
    ):
        super(Generator, self).__init__()

        # Layers for separated inputs.
        self.convT_z = ConvT(
            in_channels=noise_dim,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU(),
        )

        self.convT_c = ConvT(
            in_channels=conditional_dim,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU(),
        )

        # Layers for combined inputs.
        convT_zc = []
        for i in range(1, len(filters)):
            convT_zc.append(
                ConvT(
                    in_channels=filters[i - 1],
                    out_channels=filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    normalization=nn.BatchNorm2d,
                    activation=nn.ReLU(),
                )
            )

        # Output layer.
        convT_out = ConvT(
            in_channels=filters[-1],
            out_channels=output_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            normalization=None,
            activation=nn.Tanh(),
        )

        self.convT = nn.Sequential(*convT_zc, convT_out)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        out_z = self.convT_z(z)
        out_c = self.convT_c(c)
        # Combine input and conditional informations.
        out = torch.cat((out_z, out_c), dim=1)
        out = self.convT(out)

        print(out.shape)


if __name__ == "__main__":
    image_size = 32
    label_dim = 10
    G_input_dim = 100
    G_output_dim = 1
    D_input_dim = 1
    D_output_dim = 1
    num_filters = [512, 256, 128]

    G = Generator(G_input_dim, label_dim, num_filters, G_output_dim)
    # D = Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)

    # x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, label_dim, (32,))
    z = torch.randn(32, G_input_dim, 1, 1)
    c = F.one_hot(y, label_dim).view(32, label_dim, 1, 1).to(torch.float)

    out = G(z, c)
