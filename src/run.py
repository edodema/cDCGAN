import torch
import torchvision
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path("..")
    DATA = ROOT / "data"
    print(DATA.exists())
