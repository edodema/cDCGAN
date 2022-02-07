from src.common.opt import Options
from src.common.utils import get_torch_dataset, get_stats
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.model.gan import Discriminator, Generator


def train(
    opt,
    image_size: int,
    num_classes: int,
    noise_size: int,
    D: nn.Module,
    G: nn.Module,
    device: torch.device,
):
    G = G.to(device)
    D = D.to(device)
    # Data.
    dataset = get_torch_dataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    # Criterion.
    loss_fn = nn.BCELoss()

    # Optimizers.
    D_opt = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    G_opt = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    D_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        D_opt, milestones=[5, 10], gamma=0.1
    )
    G_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        G_opt, milestones=[5, 10], gamma=0.1
    )

    # Label preprocess.
    onehot = (
        torch.zeros((num_classes, num_classes), device=device)
        .scatter(1, torch.arange(num_classes, device=device).view(num_classes, 1), 1)
        .view(num_classes, num_classes, 1, 1)
    )

    fill = torch.zeros(
        (num_classes, num_classes, image_size, image_size), device=device
    )
    for i in range(num_classes):
        fill[i, i, :, :] = 1

    # Train cycle.
    step = 0
    for epoch in range(opt.epochs):
        D_losses = []
        G_losses = []

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            c = fill[y]

            # Train discriminator with real data.
            D_real = D(x, c)
            # print(D_real.shape)


if __name__ == "__main__":
    opt = Options().parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.train:
        # Train.
        if opt.dataset == "cifar10":
            img_size = 32
            num_classes = 10
            noise_dim = 100
            filters = [1024, 512, 256, 128]

            d = Discriminator(
                in_channels=3,
                conditional_dim=num_classes,
                filters=filters[::-1],
                out_channels=1,
            )
            g = Generator(
                noise_dim=noise_dim,
                conditional_dim=num_classes,
                filters=filters,
                out_channels=3,
            )
        elif opt.dataset == "mnist":
            img_size = 28
            cond_dim = 10
            # noise_dim = 100
            # filters = [1024, 512, 256, 128]

            # d = Discriminator(
            #     in_channels=1,
            #     conditional_dim=cond_dim,
            #     filters=filters[::-1],
            #     out_channels=1,
            # )
            # g = Generator(
            #     noise_dim=noise_dim,
            #     conditional_dim=cond_dim,
            #     filters=filters,
            #     out_channels=1,
            # )
        else:
            raise Exception("Train for this dataset has not been implemented yer.")

        # Train.
        # ! Create optimizer.

        train(
            opt,
            image_size=img_size,
            num_classes=num_classes,
            noise_size=noise_dim,
            D=d,
            G=g,
            device=device,
        )

    # CelebA dataset
    # transform = transforms.Compose(
    #     [
    #         transforms.Scale(image_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #     ]
    # )

    # celebA_data = datasets.ImageFolder(data_dir, transform=transform)
    # celebA_data.imgs.sort()

    # data_loader = torch.utils.data.DataLoader(
    #     dataset=celebA_data, batch_size=batch_size, shuffle=False
    # )
