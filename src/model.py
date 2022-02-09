import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, channels: int, conditional_dim: int, img_size: int = 64):
        """Discriminator.

        Args:
            channels (int): Input channels.
            img_size (int, optional): Image size. Defaults to 64.
        """
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=channels + conditional_dim,
                out_channels=img_size,
                kernel_size=[4, 4],
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            self.block(in_channels=img_size, out_channels=img_size * 2),
            self.block(in_channels=img_size * 2, out_channels=img_size * 4),
            self.block(in_channels=img_size * 4, out_channels=img_size * 8),
            nn.Conv2d(
                in_channels=img_size * 8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Module:
        """Default block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int, optional): Kernel size. Defaults to 4.
            stride (int, optional): Stride. Defaults to 2.
            padding (int, optional): Padding. Defaults to 1.

        Returns:
            nn.Module: Output block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.cat((x, c.to(torch.float)), dim=dim)
        return self.net(x)

    def init(self):
        """Initialize weights."""

        def weights_init(m: nn.Module):
            """Initialize weights.

            Args:
                m (nn.Module): A module.
            """
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

        self.apply(weights_init)


class Generator(nn.Module):
    def __init__(
        self, noise_dim: int, conditional_dim: int, channels: int, img_size: int = 64
    ):
        """Generator.

        Args:
            noise_dim (int): Noise vector size.
            conditional_dim (int): Conditional vector size.
            channels (int): Generated image channels.
            img_size (int, optional): Generated image size. Defaults to 64.
        """
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            self.block(noise_dim + conditional_dim, img_size * 16, stride=1, padding=0),
            self.block(img_size * 16, img_size * 8),
            self.block(img_size * 8, img_size * 4),
            self.block(img_size * 4, img_size * 2),
            nn.ConvTranspose2d(
                in_channels=img_size * 2,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Module:
        """Default block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int, optional): Kernel size. Defaults to 4.
            stride (int, optional): Stride. Defaults to 2.
            padding (int, optional): Padding. Defaults to 1.

        Returns:
            nn.Module: Output block.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Forward pass.

        Args:
            z (torch.Tensor): Noise vector.
            c (torch.Tensor): Conditional vector.
            dim (int, optional): Concatenation axis. Defaults to 1.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.cat((z, c.to(torch.float)), dim=dim)
        return self.net(x)

    def init(self):
        """Initialize weights."""

        def weights_init(m: nn.Module):
            """Initialize weights.

            Args:
                m (nn.Module): A module.
            """
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

        self.apply(weights_init)
