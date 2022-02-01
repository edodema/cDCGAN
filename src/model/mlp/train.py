from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from src.model.mlp.gan import Discriminator, Generator
import wandb

# ? This will probably be deleted.

if __name__ == "__main__":
    ROOT = Path(".")

    # Variables.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    lr = 2e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Size of the conditional vector.
    conditional_size = 10

    # Need more than 100 epochs for training generator
    epochs = 30
    step = 0
    # Train the generator k more steps than the discriminator.
    k = 1
    noise_size = 100
    num_classes = 10

    img_shape = [1, 28, 28]

    # Loss function.
    loss_fn = nn.BCEWithLogitsLoss()

    # Dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    mnist = torchvision.datasets.MNIST(
        root=ROOT, train=True, transform=transform, download=True
    )

    # Discriminator label to real
    gt_real = torch.ones([batch_size, 1]).to(device)
    # Discriminator label to fake
    gt_fake = torch.zeros([batch_size, 1]).to(device)

    # GAN.
    d = Discriminator(
        in_size=784,
        conditional_size=10,
        features=[512, 256, 1],
        norm_layers=[None] * 3,
        activations=[nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.Sigmoid()],
        dropouts=[0] * 3,
    )

    g = Generator(
        in_size=100,
        conditional_size=10,
        features=[128, 256, 512, 1024, 784],
        norm_layers=[
            None,
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(1024),
            None,
        ],
        activations=[
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            None,
        ],
        dropouts=[0] * 5,
    )

    # Dataloader.
    data_loader = DataLoader(
        dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Optimizer.
    opt_d = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))

    # Validation.
    val_z = torch.randn(size=(8 * num_classes, noise_size), device=device)

    val_c = F.one_hot(
        torch.arange(end=8 * num_classes, device=device) % num_classes,
        num_classes=num_classes,
    )

    # Logging.
    wandb.login()
    wandb.init(project="MLP", entity="edodema")
    wandb.config = {"learning_rate": lr, "epochs": epochs, "batch_size": batch_size}

    # Training.
    for epoch in range(epochs):
        for idx, (images, labels) in enumerate(data_loader):
            # Training Discriminator
            x = images.to(device)
            x = x.view(x.shape[0], -1)

            y = labels.view(
                batch_size,
            )
            c = F.one_hot(y, num_classes).to(device)
            p_real = d(x, c)
            loss_real = loss_fn(p_real, gt_real)

            z = torch.randn(batch_size, noise_size).to(device)
            p_fake = d(g(z, c), c)
            loss_fake = loss_fn(p_fake, gt_fake)
            loss_d = loss_real + loss_fake

            d.zero_grad()
            loss_d.backward()
            opt_d.step()

            wandb.log({"discriminator/loss_real": loss_real})
            wandb.log({"discriminator/loss_fake": loss_fake})
            wandb.log({"discriminator/loss": loss_d})

            if step % k == 0:
                # Training Generator
                z = torch.randn(batch_size, noise_size).to(device)
                p_fake = d(g(z, c), c)
                loss_g = loss_fn(p_fake, gt_fake)

                wandb.log({"generator/loss": loss_g})

                g.zero_grad()
                loss_g.backward()
                opt_g.step()

            # if step % 500 == 0:
            #     print(
            #         "Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}".format(
            #             epoch, epochs, step, loss_d.item(), loss_g.item()
            #         )
            #     )

            # Plot images.
            c, h, w = img_shape
            imgs = g(val_z, val_c)
            imgs = imgs.reshape(imgs.shape[0], c, h, w)
            print(imgs.shape)
            grid = torchvision.utils.make_grid(imgs, nrow=num_classes)
            print(grid.shape)
            print()
            wandb.log({"examples": grid})
