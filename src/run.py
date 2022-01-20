import torch
from torch.utils.data import DataLoader
from src.common.utils import get_dataset
from src.model.swin import SwinTransformer

if __name__ == "__main__":
    ds = get_dataset(root="../data", name="mnist", download=False)
    data_loader = DataLoader(dataset=ds, batch_size=16, shuffle=True, drop_last=False)
    # model = SwinTransformer(
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
    #
    # print(model)

    for xb in data_loader:
        x = xb[0]
        y = xb[1]

        # out = model(x)

        print(f"x: {x.shape}")
        # print(f"out: {out}")
        # print(torch.mean(x))
        # print(torch.std(x))
        break
