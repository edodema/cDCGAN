import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="cDCGAN", description="Usage.", add_help=True
        )
        self.opt = None

    def _init(self):
        # Add description

        # * Directories.
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="./data",
            help="Directory in which data are stored.",
        )

        self.parser.add_argument(
            "--save_dir",
            type=str,
            default="./outputs",
            help="Directory in which outputs are saved.",
        )

        # * Dataset.
        self.parser.add_argument(
            "--dataset",
            type=str,
            choices=[
                "celeb_a",
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

        self.parser.add_argument(
            "--no-download",
            dest="download",
            action="store_false",
            default=True,
            help="Do not download the dataset, true by default.",
        )
        self.parser.set_defaults(download=True)

        self.parser.add_argument(
            "--no-normalize",
            dest="normalization",
            action="store_false",
            help="Do not normalize data by z-score, true by default.",
        )
        self.parser.set_defaults(normalization=True)

        # * Training.
        self.parser.add_argument(
            "--train", dest="train", action="store_true", help="Train GAN."
        )
        self.parser.set_defaults(train=False)

    def _print(self):
        pprint(vars(self.opt), indent=4)

    def parse(self):
        self._init()
        self.opt = self.parser.parse_args()
        return self.opt
