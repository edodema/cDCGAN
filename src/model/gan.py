import torch
from torch import nn
from swin import SwinTransformer


class Discriminator(nn.Module):
    def __init__(self, input_size: int, condition_size: int, num_classes: int = 1):
        super(Discriminator, self).__init__()
        self.swin = SwinTransformer(
            img_size=28,
            patch_size=2,
            in_chans=1,
            num_classes=10,
            embed_dim=15,
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

    def forward(self, x, c):
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self):
        pass


if __name__ == "__main__":
    pass
