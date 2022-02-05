from pathlib import Path
from git import tag
import torch
import torch.nn as nn
import torchvision
from src.data.datamodule import TorchDataModule
from src.model.cnn.gan import LitGAN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    ROOT = Path(".")

    # Configuration.
    cfg_data = {
        "name": "cifar10",
        "root": ROOT / "data",
        "download": False,
        "transforms": [],
        "num_workers": 12,
        "batch_size": 64,
        "shuffle": True,
    }

    cfg_d = {
        "in_channels": 1,
        "conditional_size": 10,
        "features": [32, 16, 16, 12, 11, 10],
        "kernels": [3, 5, 1, 3, 5, 3],
        "strides": [1, 2, 1, 1, 2, 1],
        "paddings": [0] * 6,
        "norm_layers": [
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(16),
            nn.BatchNorm2d(16),
            nn.BatchNorm2d(12),
            nn.BatchNorm2d(11),
            nn.BatchNorm2d(10),
        ],
        "activations": [nn.ReLU()] * 6,
        "loss": "BCE",
        "lr": 1e-3,
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
        "norm_layers": [
            nn.BatchNorm2d(5),
            nn.BatchNorm2d(3),
            nn.BatchNorm2d(5),
            nn.BatchNorm2d(3),
            nn.BatchNorm2d(1),
        ],
        "activations": [nn.ReLU()] * 5,
        "loss": "BCE",
        "lr": 1e-3,
    }

    cfg_gan = {"d": cfg_d, "g": cfg_g, "num_classes": 10}

    cfg_train = {
        "deterministic": True,
        "random_seed": 42,
        "val_check_interval": 1.0,
        "progress_bar_refresh_rate": 20,
        "fast_dev_run": False,  # True for debug purposes.
        "gpus": -1 if torch.cuda.is_available() else 0,
        "precision": 32,
        # "max_steps": 10,
        "max_epochs": 5,
        "accumulate_grad_batches": 1,
        "num_sanity_val_steps": 2,
        "gradient_clip_val": 10.0,
    }

    if cfg_train["deterministic"]:
        pl.seed_everything(cfg_train["random_seed"])

    # Instantiate.
    # wandb_logger = WandbLogger(project="CNN", tags=["try"])
    datamodule = TorchDataModule(cfg_data)
    # model = LitGAN(cfg_gan)

    # trainer = pl.Trainer(
    #     deterministic=cfg_train["deterministic"],
    #     gpus=cfg_train["gpus"],
    #     max_epochs=cfg_train["max_epochs"],
    #     logger=wandb_logger,
    #     # callbacks=[checkpoint_callback],
    #     # log_every_n_steps=1
    # )

    # # Fit.
    # trainer.fit(model=model, datamodule=datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.ds_train[0][0].shape)
