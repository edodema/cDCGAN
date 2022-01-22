from black import out
import torch
from torch import nn
from src.model.swin import SwinTransformer
from src.common.utils import get_dataset
from torch.utils.data import DataLoader
from pathlib import Path


class Discriminator(nn.Module):
    def __init__(self, input_size: int, condition_size: int):
        super(Discriminator, self).__init__()

        self.condition_size = condition_size

        self.swin = SwinTransformer(
            mode="d",
            img_size=input_size,
            patch_size=2,
            in_chans=1,
            num_classes=1,
            embed_dim=15,  # TODO: I could add the condition size in the embedding.
            depths=[2],
            num_heads=[3],
            window_size=1,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )

    def forward(self, x):
        # For now we ignore the class c, thus is not conditional.
        out = self.swin(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, condition_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size

        self.condition_size = condition_size
        self.output_size = output_size

        self.swin = SwinTransformer(
            mode="g",
            img_size=input_size,
            patch_size=2,
            in_chans=1,
            num_classes=output_size * output_size,
            embed_dim=15,  # TODO: I could add the condition size in the embedding.
            depths=[2],
            num_heads=[3],
            window_size=1,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )

    def forward(self, z):
        out = self.swin(z)
        return out.view(z.shape)


if __name__ == "__main__":
    ROOT = Path(".")
    ds = get_dataset(root=ROOT / "data", name="mnist", download=False)
    data_loader = DataLoader(dataset=ds, batch_size=16, shuffle=True, drop_last=False)

    # d =

    # g =

    for xb in data_loader:
        x = xb[0]
        y = xb[1]
        # print(f"x: {x.shape}")
        # print(f"y: {y.shape}")

        # out = d(x, y)
        # print(f"d: {out.shape}")

        # out = g(z=torch.rand(x.shape[0], 100), c=y)
        # print(f"g: {out.shape}")
        break
