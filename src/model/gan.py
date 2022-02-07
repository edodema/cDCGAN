from typing import *
import torch
from torch import nn
import torch.nn.functional as F

# * Basic modules.


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        normalization: nn.Module = nn.Identity,
        activation: nn.Module = nn.Identity(),
    ):
        """Convolutional block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride value.
            padding (int): Padding value.
            normalization (nn.Module): Noralization we want to use. Defaults to nn.Identity.
            activation (nn.Module): Activation function, it should be already instantiated. Defaults to nn.Identity().
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[kernel_size, kernel_size],
            stride=stride,
            padding=padding,
        )

        # In GAN's seminal paper is advised to use a normal distribution.
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(self.conv.bias, val=0.0)

        self.norm = normalization(out_channels)

        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        normalization: nn.Module = nn.Identity,
        activation: nn.Module = nn.Identity(),
    ):
        """Transposed convolution block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride value.
            padding (int): Padding value.
            normalization (nn.Module): Noralization we want to use. Defaults to nn.Identity.
            activation (nn.Module): Activation function, it should be already instantiated. Defaults to nn.Identity().
        """
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

        self.norm = normalization(out_channels)

        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.convT(x)
        out = self.norm(out)
        out = self.act(out)
        return out


# * Generative adversarial network.


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conditional_dim: int,
        filters: List[int],
        out_channels: int,
    ):
        """Discriminator.

        Args:
            noise_dim (int): Number of input channels.
            conditional_dim (int): Dimension of the conditional vector.
            filters (List[int]): List of feature map dimensions.
            out_channels (int): Output image channels.
        """
        super(Discriminator, self).__init__()

        # Layers for separated inputs.
        self.conv_x = Conv(
            in_channels=in_channels,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            activation=nn.LeakyReLU(0.2),
        )

        self.conv_c = Conv(
            in_channels=conditional_dim,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            activation=nn.LeakyReLU(0.2),
        )

        # Layers for combined inputs.
        conv_xc = []
        for i in range(1, len(filters)):
            conv_xc.append(
                Conv(
                    in_channels=filters[i - 1],
                    out_channels=filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    normalization=nn.BatchNorm2d,
                    activation=nn.LeakyReLU(0.2),
                )
            )

        # Output layer.
        conv_out = Conv(
            in_channels=filters[-1],
            out_channels=out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            activation=nn.Sigmoid(),
        )

        self.conv = nn.Sequential(*conv_xc)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditional tensor.

        Returns:
            torch.Tensor: Probability distribution over real/fake output.
        """
        out_x = self.conv_x(x)
        out_c = self.conv_c(c)
        # Combine input and conditional informations.
        out = torch.cat((out_x, out_c), dim=1)
        out = self.conv(out)
        # Reshape to b, output_dim.
        # out = out.view(out.shape[0], out.shape[1])
        return out


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        conditional_dim: int,
        filters: List[int],
        out_channels: int,
    ):
        """Generator.

        Args:
            noise_dim (int): Dimension of the sampled noise.
            conditional_dim (int): Dimension of the conditional vector.
            filters (List[int]): List of feature map dimensions.
            out_channels (int): Output image channels.
        """
        super(Generator, self).__init__()

        # Layers for separated inputs.
        self.convT_z = ConvT(
            in_channels=noise_dim,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=1,
            padding=0,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU(),
        )

        self.convT_c = ConvT(
            in_channels=conditional_dim,
            out_channels=filters[0] // 2,
            kernel_size=4,
            stride=1,
            padding=0,
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
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            activation=nn.Tanh(),
        )

        self.convT = nn.Sequential(*convT_zc, convT_out)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z (torch.Tensor): Noise tensor.
            c (torch.Tensor): Conditional tensor.

        Returns:
            torch.Tensor: Generated image.
        """
        out_z = self.convT_z(z)
        out_c = self.convT_c(c)
        # Combine input and conditional informations.
        out = torch.cat((out_z, out_c), dim=1)
        out = self.convT(out)
        return out


if __name__ == "__main__":
    image_size = 32
    label_dim = 10
    G_input_dim = 100
    G_output_dim = 3
    D_input_dim = 3
    D_output_dim = 1
    num_filters = [1024, 512, 256, 128]

    G = Generator(G_input_dim, label_dim, num_filters, G_output_dim)
    D = Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)

    y = torch.randint(0, label_dim, (32,))
    z = torch.randn(32, G_input_dim, 1, 1)
    c = F.one_hot(y, label_dim).view(32, label_dim, 1, 1).to(torch.float)

    out = G(z, c)
    print("G ", out.shape)

    x = torch.randn(32, 3, image_size, image_size)
    # onehot = (
    #     torch.zeros(label_dim, label_dim)
    #     .scatter(1, torch.arange(label_dim).view(label_dim, 1), 1)
    #     .view(label_dim, label_dim, 1, 1)
    # )
    fill = torch.zeros(label_dim, label_dim, image_size, image_size)
    for i in range(label_dim):
        fill[i, i, :, :] = 1

    labels = torch.randint(0, label_dim, (32,))
    c_fill_ = fill[labels]

    out = D(x, c_fill_)
    print("D ", out.shape)
