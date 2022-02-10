from ast import arg
from distutils.command.config import config
from typing import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from src.parser import args
from src.utils import get_torch_dataset, display
from src.model import Discriminator, Generator
from tqdm.auto import tqdm
import wandb


def train(device: torch.device):
    """Train function.

    Args:
        device (torch.device): Device.
    """
    # Load data.
    dataset = get_torch_dataset(
        opt=args,
        transform=[torchvision.transforms.Resize((args.img_size, args.img_size))],
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
        torch.arange(end=args.ncol * args.num_classes, device=device)
        % args.num_classes,
        num_classes=args.num_classes,
    )[..., None, None]

    # Mean losses that will be displayed during training each delta steps.
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    delta_step = len(dataloader)

    if args.wandb:
        wandb.watch(
            models=(d, g),
            criterion=loss_fn,
            log="all",
            log_freq=1,
            idx=None,
            log_graph=(True),
        )
    step = 0
    for i, epoch in enumerate(range(args.epochs)):
        # Minibatch training.
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Number of examples, that number will not necessarily be equal to batch_size.
            num_examples = images.shape[0]
            oh = F.one_hot(labels, args.num_classes).unsqueeze(-1).unsqueeze(-1)

            # * Train the discriminator.
            # Genere false images.
            imgs_fake = g(
                z=torch.randn((num_examples, args.noise_dim, 1, 1), device=device), c=oh
            )

            # Convert oh of shape batch_size x num_classes x 1 x 1 to batch_size x num_classes x img_size x img_size.
            # Basically convert it to a one hot encoded 4D array such that only the correct class 2D array has all 1s
            # and the rest have 0s in them for every image. The discriminator needs a different shape for oh.
            oh_d = oh.clone().repeat(1, 1, args.img_size, args.img_size)

            d_opt.zero_grad()

            preds_fake = d(x=imgs_fake.detach(), c=oh_d)

            # The target is the truth values for fake images i.e. 0.
            d_loss = loss_fn(input=preds_fake, target=torch.zeros_like(preds_fake))

            # Probabilities assigned to real images, we do not need to detach since we minimize on real images.
            preds_real = d(x=images, c=oh_d)
            # The target is the truth values for real images i.e. 1.
            d_loss += loss_fn(input=preds_real, target=torch.ones_like(preds_real))

            # Average them.
            d_loss /= 2

            if args.wandb:
                wandb.log({"d_loss": d_loss})

            d_loss.backward()
            d_opt.step()

            # * Train the generator.
            g_opt.zero_grad()

            z = torch.randn((num_examples, args.noise_dim, 1, 1), device=device)
            imgs_new = g(z=z, c=oh)
            # Predictions on generated data.
            preds = d(x=imgs_new, c=oh_d)

            g_loss = loss_fn(input=preds, target=torch.ones_like(preds))

            if args.wandb:
                wandb.log({"g_loss": g_loss})

            g_loss.backward()
            g_opt.step()

            # Early stopping.
            if args.monitor:
                if i == 0:
                    best_d_loss = d_loss
                    best_g_loss = g_loss
                else:
                    if d_loss < best_d_loss:
                        best_d_loss = d_loss
                        print(f"Found better discriminator: {best_d_loss}")
                    if g_loss < best_g_loss:
                        best_g_loss = g_loss
                        # Save model.
                        save_path = f"{args.checkpoint_dir}/{args.dataset}.pth"
                        torch.save(g.state_dict(), save_path)
                        print(f"Found better generator: {best_g_loss}")
                        print(f"Saved model to {save_path}")

            # Logging.
            if args.wandb:
                wandb.log({"d_loss": d_loss})
                wandb.log({"best_d_loss": best_d_loss})

            # For displaying loss after 'd_s' steps
            mean_d_loss += d_loss.item() / delta_step
            mean_g_loss += g_loss.item() / delta_step
            if step % delta_step == 0:
                # Validation.
                z = torch.randn(
                    (args.num_classes * args.ncol, args.noise_dim, 1, 1), device=device
                )

                if args.monitor:
                    g_ = Generator(
                        noise_dim=args.noise_dim,
                        conditional_dim=args.conditional_size,
                        channels=args.channels,
                        img_size=args.img_size,
                    ).to(device)
                    load_path = f"{args.checkpoint_dir}/{args.dataset}.pth"
                    g_.load_state_dict(torch.load(load_path, map_location=device))
                else:
                    g_ = g

                out = g_(
                    z=z,
                    c=val_oh,
                )

                if args.wandb:
                    image_grid = torchvision.utils.make_grid(out, nrow=args.num_classes)
                    wandb.log({"examples": wandb.Image(image_grid)})

                else:
                    print(
                        f"Epoch: {epoch+1}, Step: {step + 1}, Discriminator loss: {mean_d_loss:.4f}, Generator loss: {mean_g_loss:.4f}"
                    )
                    display(images=out, ncol=args.ncol)

                mean_d_loss = 0
                mean_g_loss = 0

            step += 1

    if not args.monitor:
        # Save model.
        save_path = f"{args.checkpoint_dir}/{args.dataset}.pth"
        torch.save(g.state_dict(), save_path)
        print(f"Saved model to {save_path}")


def create(device: torch.device):
    """Create images.

    Args:
        device (torch.device): Device.
    """
    # Model.
    g = Generator(
        noise_dim=args.noise_dim,
        conditional_dim=args.conditional_size,
        channels=args.channels,
        img_size=args.img_size,
    )

    load_path = f"{args.checkpoint_dir}/{args.dataset}.pth"
    g.load_state_dict(torch.load(load_path, map_location=device))
    g.eval()

    # Create an image for each class.
    if args.label == -1:
        c = F.one_hot(
            torch.arange(end=args.ncol * args.num_classes, device=device)
            % args.num_classes,
            num_classes=args.num_classes,
        )[..., None, None]
        z = torch.randn(
            (args.num_classes * args.ncol, args.noise_dim, 1, 1), device=device
        )
    # Create an image for a specific class.
    else:
        c = F.one_hot(torch.tensor(args.label), num_classes=args.num_classes)[
            None, :, None, None
        ]
        z = torch.randn((1, args.noise_dim, 1, 1), device=device)

    # Display.
    imgs = g(z=z, c=c)
    display(images=imgs, ncol=args.ncol)


if __name__ == "__main__":
    # Logging.
    if args.wandb:
        entity, project = args.wandb.split("/", maxsplit=1)
        wandb.init(entity=entity, project=project, config=vars(args))

    device = torch.device(args.device)

    if args.train:
        train(device)
    else:
        create(device)

    wandb.finish()
