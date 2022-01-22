from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.common.utils import get_dataset

if __name__ == "__main__":
    ROOT = Path(".")
    ds = get_dataset(root=ROOT / "data", name="mnist", download=False)
    data_loader = DataLoader(dataset=ds, batch_size=16, shuffle=True, drop_last=False)

    # model = SwinTransformer(
    #     mode="d",
    #     img_size=28,
    #     patch_size=2,
    #     in_chans=1,
    #     num_classes=10,
    #     embed_dim=15,
    #     depths=[2],
    #     num_heads=[3],
    #     window_size=1,
    #     mlp_ratio=4.0,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     drop_path_rate=0.1,
    #     norm_layer=torch.nn.LayerNorm,
    #     ape=False,
    #     patch_norm=True,
    #     use_checkpoint=False,
    # )

    # d = Discriminator(input_size=28, condition_size=10, output_size=1)
    # g = Generator(input_size=28, condition_size=10, output_size=28)

    for xb in data_loader:
        x = xb[0]
        y = xb[1]
        # z = torch.rand(x.shape)

        # out = g(z)

        # plt.imshow(x[0].permute(1, 2, 0))
        # plt.show()

        # plt.imshow(z[0].permute(1, 2, 0))
        # plt.show()

        print(f"x: {x.shape}")
        print(f"y: {y}")
        # print(f"out: {out.shape}")
        # print(torch.mean(x))
        # print(torch.std(x))
        break
