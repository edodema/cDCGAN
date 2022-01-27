import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from src.model.mlp.mlp import MLP
from src.data.datamodule import TorchDataModule
from pathlib import Path
from src.data.datamodule import TorchDataModule
from typing import *
import pytorch_lightning as pl


class Discriminator(nn.Module):
    def __init__(
        self,
        in_size: int,
        conditional_size: int,
        features: List[int],
        norm_layers: List[int],
        activations: List[int],
        dropouts: List[int],
    ) -> None:
        """Discriminator.

        Args:
            in_size (int): Input size.
            conditional_size (int): Conditional vector size.
            features (List[int]): List of linear layers' features.
            norm_layers (List[nn.Module]): List of normalization layers.
            activations (List[nn.Module]): List of activations functions.
            dropouts (List[int]): List of dropout probabilities.
        """
        super(Discriminator, self).__init__()

        self.fc = MLP(
            in_size=in_size + conditional_size,
            features=features,
            norm_layers=norm_layers,
            activations=activations,
            dropouts=dropouts,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Batch of input images.
            c (torch.Tensor): Batch of conditioning vectors.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(torch.cat((x.view(x.shape[0], -1), c), dim=-1))


class Generator(nn.Module):
    def __init__(
        self,
        in_size: int,
        conditional_size: int,
        features: List[int],
        norm_layers: List[nn.Module],
        activations: List[nn.Module],
        dropouts: List[int],
    ):
        """Generator.

        Args:
            in_size (int): Input size.
            conditional_size (int): Conditional vector size.
            features (List[int]): List of linear layers' features.
            norm_layers (List[nn.Module]): List of normalization layers.
            activations (List[nn.Module]): List of activations functions.
            dropouts (List[int]): List of dropout probabilities.
        """
        super(Generator, self).__init__()

        self.fc = MLP(
            in_size=in_size + conditional_size,
            features=features,
            norm_layers=norm_layers,
            activations=activations,
            dropouts=dropouts,
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z (torch.Tensor): Batch of random vectors.
            c (torch.Tensor): Batch of conditioning vectors.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(torch.cat((z, c), dim=-1))


class LitGAN(pl.LightningModule):
    def __init__(self, cfg: Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # Discriminator.
        self.d = Discriminator(
            in_size=cfg["d"]["in_size"],
            conditional_size=cfg["d"]["conditional_size"],
            features=cfg["d"]["features"],
            norm_layers=cfg["d"]["norm_layers"],
            activations=cfg["d"]["activations"],
            dropouts=cfg["d"]["dropouts"],
        )

        # Generator.
        self.g = Generator(
            in_size=cfg["g"]["in_size"],
            conditional_size=cfg["g"]["conditional_size"],
            features=cfg["g"]["features"],
            norm_layers=cfg["g"]["norm_layers"],
            activations=cfg["g"]["activations"],
            dropouts=cfg["g"]["dropouts"],
        )

        self.num_classes = cfg["num_classes"]
        self.img_shape = cfg["img_shape"]

        self.g_z_len = cfg["g"]["in_size"]
        self.g_c_len = cfg["g"]["conditional_size"]

        self.lr_d = cfg["d"]["lr"]
        self.lr_g = cfg["g"]["lr"]

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

        # Generated images for validation. Torchvision has a maximum of 8 columns by default.
        cols = 8
        self.val_c = F.one_hot(
            torch.arange(end=cols * self.num_classes, device=self.device)
            % self.num_classes,
            num_classes=self.num_classes,
        )
        self.val_z = torch.randn(
            size=(cols * self.num_classes, self.g_z_len), device=self.device
        )

        # TODO: add metric

    def d_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Training step for the discriminator.

        Args:
            x (torch.Tensor): Batch of real images.
            y (torch.Tensor): Tensor of ground truth labels.

        Raises:
            Exception: Discriminator loss undefined.
            Exception: Generator loss undefined.

        Returns:
            torch.Tensor: Computed loss.
        """
        batch_size = x.shape[0]

        # Real images.
        out = self.d(x, F.one_hot(y, num_classes=self.num_classes))

        # Since images are real we expect all true.
        target_real = torch.ones(batch_size, device=self.device)

        # Real loss on data distribution.
        if isinstance(self.d_loss, nn.BCEWithLogitsLoss):
            loss_real = self.d_loss(out.squeeze(1), target_real)
        else:
            raise Exception("Discriminator loss undefined, please define it.")

        # Fake images.
        z = torch.randn(size=(batch_size, self.g_z_len), device=self.device)
        c = F.one_hot(
            torch.randint(
                low=0, high=self.num_classes, size=(batch_size,), device=self.device
            ),
            num_classes=self.num_classes,
        )
        logits = self.d(self.g(z, c), c)

        # Since images are fake we would want all false.
        target_fake = torch.zeros(batch_size, device=self.device)

        # Loss on fake images.
        if isinstance(self.g_loss, nn.BCEWithLogitsLoss):
            loss_fake = self.d_loss(logits.squeeze(1), target_fake)
        else:
            raise Exception("Discriminator loss undefined, please define it.")

        loss = loss_real + loss_fake

        # Logging.
        self.log("discriminator/loss", loss)
        self.log("discriminator/loss_real", loss_real)
        self.log("discriminator/loss_fake", loss_fake)

        return loss

    def g_step(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        # Classify generated images using the discriminator.
        logits = self.d(self.g(z, c), c)

        # We aim to maximize d's loss, it is the same as minimizing the loss with true labels flipped i.e. target = 1 for fake images.
        target = torch.ones(batch_size, device=self.device)

        if isinstance(self.g_loss, nn.BCEWithLogitsLoss):
            loss = self.g_loss(logits.squeeze(1), target)
        else:
            raise Exception("Generator loss undefined, please define it.")

        # Logging.
        self.log("generator/loss", loss)

        return loss

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        x, y = batch
        batch_size = x.shape[0]

        # Train generator.
        if optimizer_idx == 0:
            z = torch.randn(size=(batch_size, self.g_z_len), device=self.device)
            c = F.one_hot(
                torch.randint(
                    low=0, high=self.num_classes, size=(batch_size,), device=self.device
                ),
                num_classes=self.num_classes,
            )
            loss = self.g_step(z, c)

        # Train discriminator.
        if optimizer_idx == 1:
            loss = self.d_step(x, y)

        return loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.g.parameters(), lr=self.lr_g)
        d_opt = torch.optim.Adam(self.d.parameters(), lr=self.lr_d)
        return [g_opt, d_opt], []

    def on_epoch_end(self) -> None:
        val_z = self.val_z.to(self.device)
        val_c = self.val_c.to(self.device)

        c, h, w = self.img_shape
        imgs = self.g(val_z, val_c)
        imgs = imgs.reshape(imgs.shape[0], c, h, w)

        grid = torchvision.utils.make_grid(imgs)
        self.logger.log_image(key="generated_images", images=[grid])


if __name__ == "__main__":
    ROOT = Path(".")

    cfg_data = {
        "name": "mnist",
        "root": ROOT / "data",
        "download": False,
        "transforms": [],
        "num_workers": 12,
        "batch_size": 32,
        "shuffle": False,
    }

    cfg_d = {
        "in_size": 784,
        "conditional_size": 10,
        "features": [512, 256, 1],
        "norm_layers": [None] * 3,
        "activations": [nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.Sigmoid()],
        "dropouts": [0] * 3,
        "loss": "BCE",
        "lr": 1e-5,
    }

    cfg_g = {
        "in_size": 100,
        "conditional_size": 10,
        "features": [128, 256, 512, 1024, 784],
        "norm_layers": [
            None,
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(1024),
            None,
        ],
        "activations": [
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            None,
        ],
        "dropouts": [0] * 5,
        "loss": "BCE",
        "lr": 1e-5,
    }

    cfg_gan = {"d": cfg_d, "g": cfg_g, "num_classes": 10, "img_shape": [1, 28, 28]}

    ds = TorchDataModule(cfg_data)
    ds.prepare_data()
    ds.setup()
    dl = ds.train_dataloader()

    gan = LitGAN(cfg_gan)

    gan.on_epoch_end()

    for xb in dl:
        x = xb[0]
        y = xb[1]

        print(x.shape, y.shape)
        # out_g = gan.g_step(x.shape[0])

        # out_d = gan.d_step(x, y)

        # print(f"g_loss: {out_g}")
        # print(f"d_loss: {out_d}")

        break
