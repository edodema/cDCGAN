from pathlib import Path
import torch
import torch.nn as nn
from src.data.datamodule import TorchDataModule
from src.model.mlp.gan import LitGAN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    ROOT = Path(".")

    # Configuration.
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
    wandb_logger = WandbLogger(project="MLP", tags=["try"])
    datamodule = TorchDataModule(cfg_data)
    model = LitGAN(cfg_gan)

    trainer = pl.Trainer(
        deterministic=cfg_train["deterministic"],
        gpus=cfg_train["gpus"],
        max_epochs=cfg_train["max_epochs"],
        logger=wandb_logger,
        # callbacks=[checkpoint_callback],
        # log_every_n_steps=1
    )

    # Fit.
    trainer.fit(model=model, datamodule=datamodule)
