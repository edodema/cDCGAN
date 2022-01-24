import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from src.model.cnn.cnn import Conv, ConvTranspose
from src.data.datamodule import TorchDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.datamodule import TorchDataModule
from typing import *
import pytorch_lightning as pl
import matplotlib.pyplot as plt


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
        out = self.cnn(x)

        assert (
            c.shape == out.shape[:2]
        ), "The CNN should output a number of features equal to `conditional_size`."
        assert out.shape[2:] == torch.Size([1, 1]), "The CNN should reduce H, W to 1."

        out = out.view(out.shape[0], out.shape[1])

        # Combine image and conditioning data.
        out = torch.cat((out, c), dim=-1)

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
        x = torch.cat((z, c), dim=-1)

        out = self.fc(x)
        out = out.view(out.shape[0], self.c, self.h, self.w)

        out = self.cnn(out)
        return out


class GAN(pl.LightningModule):
    def __init__(self, cfg: Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # Discriminator.
        self.d = Discriminator(
            in_channels=cfg["d"]["in_channels"],
            conditional_size=cfg["d"]["conditional_size"],
            features=cfg["d"]["features"],
            kernels=cfg["d"]["kernels"],
            strides=cfg["d"]["strides"],
            paddings=cfg["d"]["paddings"],
            norm_layers=cfg["d"]["norm_layers"],
            activations=cfg["d"]["activations"],
        )

        # Generator.
        self.g = Generator(
            in_size=cfg["g"]["in_size"],
            h_size=cfg["g"]["h_size"],
            in_channels=cfg["g"]["in_channels"],
            conditional_size=cfg["g"]["conditional_size"],
            features=cfg["g"]["features"],
            kernels=cfg["g"]["kernels"],
            strides=cfg["g"]["strides"],
            paddings=cfg["g"]["paddings"],
            norm_layers=cfg["g"]["norm_layers"],
            activations=cfg["g"]["activations"],
        )

        self.num_classes = cfg["num_classes"]
        self.g_z_len = cfg["g"]["in_size"]
        self.g_c_len = cfg["g"]["conditional_size"]

        # We may have different losses, this way we can deal with it more easily.
        d_loss = cfg["d"]["loss"]
        if d_loss == "BCE":
            self.d_loss = nn.BCEWithLogitsLoss()
        else:
            raise Exception("Discriminator loss undefined, please define it.")

        g_loss = cfg["g"]["loss"]
        if g_loss == "BCE":
            self.g_loss = nn.BCEWithLogitsLoss()
        else:
            raise Exception("Generator loss undefined, please define it.")

        # TODO: add metric

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Generate an image using the generator.

        Args:
            z (torch.Tensor): Batch of random noise.
            c (torch.Tensor): Batch of conditioning vectors.

        Returns:
            torch.Tensor: Generated image.
        """
        out = self.g(z, c)
        return out

    def g_step(self, batch_size: int) -> float:
        """Training step for the generator.

        Args:
            batch_size (int): Batch size.

        Returns:
            float: Computed loss.
        """
        # Sample random noise, preferably from a gaussian distribution.
        z = torch.randn(batch_size, self.g_z_len)
        c = F.one_hot(
            torch.randint(low=0, high=self.num_classes, size=(batch_size,)),
            num_classes=self.num_classes,
        )

        # Generate images.
        x = self(z, c)

        # Classify images using the discriminator.
        logits = self.d(x, c)

        # We aim to maximize d's loss, it is the same as minimizing the loss with true labels flipped i.e. target = 1 for fake images.
        target = torch.ones(batch_size)

        if isinstance(self.g_loss, nn.BCEWithLogitsLoss):
            loss = self.g_loss(logits.squeeze(1), target)
        else:
            raise Exception("Generator loss undefined, please define it.")

        return loss

    def d_step(self, x, y, c):
        # Real images.
        out = self.d(x, c)

        # Real loss on data distribution.
        if isinstance(self.d_loss, nn.BCEWithLogitsLoss):
            loss_d = self.d_loss(out.squeeze(1), y.to(torch.float))
        else:
            raise Exception("Discriminator loss undefined, please define it.")

        # Fake images.
        # TODO: see https://github.com/jamesloyys/PyTorch-Lightning-GAN/blob/main/CGAN/cgan.py#L117


if __name__ == "__main__":
    ROOT = Path(".")

    cfg_data = {
        "name": "mnist",
        "root": ROOT / "data",
        "download": False,
        "transforms": [torchvision.transforms.RandomHorizontalFlip(p=0.5)],
        "num_workers": 12,
        "batch_size": 32,
        "shuffle": False,
    }

    cfg_d = {
        "in_channels": 1,
        "conditional_size": 10,
        "features": [32, 16, 16, 12, 11, 10],
        "kernels": [3, 5, 1, 3, 5, 3],
        "strides": [1, 2, 1, 1, 2, 1],
        "paddings": [0] * 6,
        "norm_layers": [nn.BatchNorm2d] * 6,
        "activations": [nn.ReLU] * 6,
        "loss": "BCE",
    }

    cfg_g = {
        "in_size": 100,
        "h_size": 4,
        "conditional_size": 10,
        "in_channels": 1,
        "features": [5, 3, 5, 3, 1],
        "kernels": [3, 5, 5, 3, 2],
        "strides": [2, 2, 1, 1, 1],
        "paddings": [0, 0, 0, 0, 0],
        "norm_layers": [nn.BatchNorm2d] * 5,
        "activations": [nn.ReLU] * 5,
        "loss": "BCE",
    }

    cfg = {"d": cfg_d, "g": cfg_g, "num_classes": 10}

    ds = TorchDataModule(cfg_data)
    ds.prepare_data()
    ds.setup()
    dl = ds.train_dataloader()

    gan = GAN(cfg)

    for xb in dl:
        x = xb[0]
        y = xb[1]
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")

        # gan.g_step(x.shape[0])
        c = F.one_hot(y, num_classes=10)
        gan.d_step(x, y, c)

        break

    # for xb in data_loader:
    #     x = xb[0]
    #     y = xb[1]
    #     # print(f"x: {x.shape}")
    #     # print(f"y: {y.shape}")

    #     c = F.one_hot(y, num_classes=10)
    #     out = d(x, c)
    #     print(f"d: {out.shape}")

    #     out = g(z=torch.rand(x.shape[0], 100), c=c)
    #     print(f"g: {out.shape}")
    #     break

    # d = Discriminator(
    #     in_channels=1,
    #     conditional_size=10,
    #     features=[32, 16, 16, 12, 11, 10],
    #     kernels=[3, 5, 1, 3, 5, 3],
    #     strides=[1, 2, 1, 1, 2, 1],
    #     paddings=[0] * 6,
    #     norm_layers=[nn.BatchNorm2d] * 6,
    #     activations=[nn.ReLU] * 6,
    # )

    # g = Generator(
    #     in_size=100,
    #     h_size=4,
    #     conditional_size=10,
    #     in_channels=1,
    #     features=[5, 3, 5, 3, 1],
    #     kernels=[3, 5, 5, 3, 2],
    #     strides=[2, 2, 1, 1, 1],
    #     paddings=[0, 0, 0, 0, 0],
    #     norm_layers=[nn.BatchNorm2d] * 5,
    #     activations=[nn.ReLU] * 5,
    # )

    #     c = F.one_hot(y, num_classes=10)
    #     out = d(x, c)
    #     print(f"d: {out.shape}")

    #     out = g(z=torch.rand(x.shape[0], 100), c=c)
    #     print(f"g: {out.shape}")
    #     break
