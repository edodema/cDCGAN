from torchvision import datasets, transforms

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
            transforms.Resize(64),
        ]
    ),
    download=True,
)
