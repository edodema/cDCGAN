from argparse import ArgumentParser

import torch

from src.common.utils import get_dataset, get_stats
from pathlib import Path
from torch.utils.data import DataLoader


def parse():
    # Add description
    parser = ArgumentParser(prog="SwinGAN", description="Usage.", add_help=True)

    parser.add_argument(
        "--data", type=str, default="data", help="Directory in which data are stored."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "cifar10",
            "cifar100",
            "fashion-mnist",
            "imagenet",
            "mnist",
            "omniglot",
        ],
        default=None,
        help="Specify the name of the dataset we are working with.",
    )

    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not download the dataset, true by default.",
    )
    parser.set_defaults(download=True)

    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not normalize data by z-score, true by default.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    ds = get_dataset(root=args.data, name=args.dataset, download=args.download)
    data_loader = DataLoader(
        dataset=ds, batch_size=50000, shuffle=True, drop_last=False
    )

    for xb in data_loader:
        x = xb[0]
        y = xb[1]
        print(torch.mean(x))
        print(torch.std(x))
        break
