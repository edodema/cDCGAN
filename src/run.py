from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from src.data.datamodule import TorchDataModule
from src.model.cnn import GAN

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
        # print(f"x: {x.shape}")
        # print(f"y: {y.shape}")

        # out_g = gan.g_step(x.shape[0])
        out_d = gan.d_step(x, y)

        print(out_d)

        break
