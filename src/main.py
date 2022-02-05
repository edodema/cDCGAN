from src.common.opt import Options
from src.train import main as train
from src.common.utils import get_torch_dataset

if __name__ == "__main__":
    opt = Options().parse()

    # image_size = 64
    # label_dim = 2
    # G_input_dim = 100
    # G_output_dim = 3
    # D_input_dim = 3
    # D_output_dim = 1
    # num_filters = [1024, 512, 256, 128]

    # learning_rate = 0.0002
    # betas = (0.5, 0.999)
    # batch_size = 128
    # num_epochs = 20

    dataset_fn, def_transforms = get_torch_dataset(name=opt.dataset)

    if opt.train:
        train(opt)

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
