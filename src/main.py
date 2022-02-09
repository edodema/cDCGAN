from typing import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from src.parser import args
from src.utils import get_torch_dataset
from src.model import Discriminator, Generator
from tqdm.auto import tqdm


def train(device: torch.device):
    img_size = args.img_size
    img_channels = args.channels
    num_classes = args.num_classes
    noise_dim = args.noise_dim
    conditional_size = args.conditional_size

    # Load data.
    dataset = get_torch_dataset(
        opt=args, transform=[torchvision.transforms.Resize((img_size, img_size))]
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Model.
    d = Discriminator(
        channels=args.channels,
        conditional_dim=args.conditional_size,
        img_size=args.img_size,
    ).to(device)

    g = Generator(
        noise_dim=args.noise_dim,
        conditional_dim=args.conditional_size,
        channels=args.channels,
        img_size=args.img_size,
    ).to(device)

    if args.init_weights:
        d.init()
        g.init()

    # Optimizer.
    g_opt = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    # The number of images we want to check for in each class.
    val_oh = F.one_hot(
        torch.arange(end=args.ncol * num_classes, device=device) % num_classes,
        num_classes=num_classes,
    )[..., None, None]

    # Mean losses that will be displayed during training each delta steps.
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    delta_step = len(dataloader)

    step = 0
    for epoch in range(args.epochs):
        # Minibatch training.
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Number of examples, that number will not necessarily be equal to batch_size.
            num_examples = images.shape[0]
            oh = F.one_hot(labels, num_classes).unsqueeze(-1).unsqueeze(-1)

            # * Train the discriminator.
            # Genere false images.
            imgs_fake = g(
                z=torch.randn((num_examples, noise_dim, 1, 1), device=device), c=oh
            )

            # Convert oh of shape batch_size x num_classes x 1 x 1 to batch_size x num_classes x img_size x img_size.
            # Basically convert it to a one hot encoded 4D array such that only the correct class 2D array has all 1s
            # and the rest have 0s in them for every image. The discriminator needs a different shape for oh.
            oh_d = oh.clone().repeat(1, 1, img_size, img_size)

            d_opt.zero_grad()

            preds_fake = d(x=imgs_fake.detach(), c=oh_d)

            # The target is the truth values for fake images i.e. 0.
            d_loss = loss_fn(input=preds_fake, target=torch.zeros_like(preds_fake))

            # Probabilities assigned to real images, we do not need to detach since we minimize on real images.
            # ! Fix this.
            preds_real = d(x=images, c=oh_d)
            # # The target is the truth values for real images i.e. 1.
            # d_loss += loss_fn(input=preds_real, target=torch.ones_like(preds_real))

            # # Average them.
            # d_loss /= 2
            # d_loss.backward()
            # d_opt.step()

            # # * Train the generator.
            # g_opt.zero_grad()

            # z = torch.randn((num_examples, noise_dim, 1, 1), device=device)
            # imgs_new = g(z=z, c=oh)
            # # Predictions on generated data.
            # preds = d(x=imgs_new, c=oh_d)

            # g_loss = loss_fn(input=preds, target=torch.ones_like(preds))
            # g_loss.backward()
            # g_opt.step()

            # print(d_loss, g_loss)
            # break
        break


def eval(device: torch.device):
    pass


if __name__ == "__main__":
    device = torch.device(args.device)

    if args.train:
        train(device)
    else:
        eval(device)
