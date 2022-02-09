import argparse
import torch

parser = argparse.ArgumentParser(
    prog="cDGAN", description="Command line arguments.", add_help=True
)

# * Directories.
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Directory in which data are stored.",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="./outputs",
    help="Directory in which outputs are saved.",
)

# * Dataset.
parser.add_argument(
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

parser.add_argument(
    "--no-download",
    dest="download",
    action="store_false",
    default=True,
    help="Do not download the dataset, true by default.",
)
parser.set_defaults(download=True)

parser.add_argument(
    "--no-normalize",
    dest="normalization",
    action="store_false",
    help="Do not normalize data by z-score, true by default.",
)
parser.set_defaults(normalization=True)

# * Data.
parser.add_argument(
    "--img_size", type=int, default=64, help="Size we want to reshape our images to."
)

parser.add_argument("--channels", type=int, default=3, help="Number of channels.")

parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")

parser.add_argument("--noise_dim", type=int, default=64, help="Noise vector dimension.")

parser.add_argument(
    "--conditional_size", type=int, default=10, help="Conditional vector size."
)

parser.add_argument(
    "--ncol", type=int, default=8, help="Number of columns in validation's image grid."
)

# * Training.
parser.add_argument("--train", dest="train", action="store_true", help="Train GAN.")
parser.set_defaults(train=False)

parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")

parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")

parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")

parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")

parser.add_argument(
    "--init_weights",
    dest="init_weights",
    action="store_true",
    help="Initialize model's weights.",
)
parser.set_defaults(init_weights=False)

parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)

# Arguments.
args = parser.parse_args()
