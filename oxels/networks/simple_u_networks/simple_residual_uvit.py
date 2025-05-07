import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_downsample import SimpleDownSample
from simple_upsample import SimpleUpSample
from simple_residual_block import SimpleResidualBlock
from simple_transformer_block import SimpleTransformerBlock
from simple_norm import SimpleNorm

class SimpleResidualUViT(nn.Module):
    """
    Residual U-ViT without time embeddings, using SimpleNorm.

    Key points:
      1) One skip per stage: pos_skips after res blocks, neg_skips after downsample.
      2) Identity initialization: residual blocks and transformers start as zeros.

    Architecture per stage i:
      - ResDown: num_res_down_blocks[i] x SimpleResidualBlock(has_skip=False)
      - Store pos_skip = h
      - Down: SimpleDownSample
      - Store neg_skip = h
    Middle:
      - h = h + pos_emb
      - for each transformer: h = h + transformer(h)
    Up path per stage i (reversed):
      - h = h - neg_skip
      - h = SimpleUpSample(h)
      - h = h + pos_skip
      - for each res_up_block: h = SimpleResidualBlock(has_skip=False)(h)
    Final: SimpleNorm -> SiLU -> Conv2d(zero-init)
    """
    def __init__(
        self,
        height: int,
        width: int,
        channels_of_stage: list,
        in_channels: int = None,
        out_channels: int = 1,
        num_res_down_blocks: list = None,
        num_res_up_blocks: list = None,
        residual_dropout: list = None,
        num_transformer_blocks: int = 4,
        num_heads: int = 4,
        transformer_expansion: int = 4,
        transformer_dropout: float = 0.0,
        norm_groups: int = 8
    ):
        super().__init__()
        L = len(channels_of_stage)
        if in_channels is None:
            in_channels = channels_of_stage[0]
        if num_res_down_blocks is None:
            num_res_down_blocks = [2] * L
        if num_res_up_blocks is None:
            num_res_up_blocks = [2] * L
        if residual_dropout is None:
            residual_dropout = [0.0] * L

        # Initial conv
        self.initial_emb = nn.Conv2d(
            in_channels, channels_of_stage[0], kernel_size=3, padding=1, bias=True
        )
        nn.init.xavier_uniform_(self.initial_emb.weight, gain=1.0)
        nn.init.zeros_(self.initial_emb.bias)

        # Down path
        self.res_down = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        for i in range(L):
            blocks = nn.ModuleList()
            for _ in range(num_res_down_blocks[i]):
                blocks.append(SimpleResidualBlock(
                    out_channels=channels_of_stage[i],
                    activation=nn.SiLU,
                    has_skip=False,
                    dropout=residual_dropout[i]
                ))
            self.res_down.append(blocks)
            out_ch = channels_of_stage[i+1] if i < L-1 else channels_of_stage[-1]
            self.down_ops.append(SimpleDownSample(channels_of_stage[i], out_ch))

        # Positional embedding
        mid_h = height // (2 ** L)
        mid_w = width  // (2 ** L)
        self.pos_emb = nn.Parameter(torch.randn(1, channels_of_stage[-1], mid_h, mid_w) * 0.01)

        # Transformer blocks
        self.transformers = nn.ModuleList([
            SimpleTransformerBlock(
                channel_dim=channels_of_stage[-1],
                expansion=transformer_expansion,
                num_heads=num_heads,
                dropout=transformer_dropout
            ) for _ in range(num_transformer_blocks)
        ])

        # Up path
        self.up_ops = nn.ModuleList()
        self.res_up = nn.ModuleList()
        for i in reversed(range(L)):
            in_ch = channels_of_stage[i+1] if i < L-1 else channels_of_stage[-1]
            self.up_ops.append(SimpleUpSample(in_ch, channels_of_stage[i]))
            blocks = nn.ModuleList()
            for _ in range(num_res_up_blocks[i]):
                blocks.append(SimpleResidualBlock(
                    out_channels=channels_of_stage[i],
                    activation=nn.SiLU,
                    has_skip=False,
                    dropout=residual_dropout[i]
                ))
            self.res_up.append(blocks)

        # Final projection
        self.norm_out = SimpleNorm(
            channel_dim=channels_of_stage[0],
            method='group',
            groups=min(norm_groups, channels_of_stage[0]),
            center=True,
            scale=True
        )
        self.act_out  = nn.SiLU()
        self.conv_out = nn.Conv2d(channels_of_stage[0], out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.initial_emb(x)
        pos_skips = []
        neg_skips = []
        # Down path
        for blocks, down in zip(self.res_down, self.down_ops):
            for blk in blocks:
                h = blk(h)
            pos_skips.append(h)
            h = down(h)
            neg_skips.append(h)

        # Middle
        h = h + self.pos_emb
        for t in self.transformers:
            h = t(h)

        # Up path
        for up, blocks in zip(self.up_ops, self.res_up):
            h = h - neg_skips.pop()
            h = up(h)
            h = h + pos_skips.pop()
            for blk in blocks:
                h = blk(h)

        # Final projection
        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)


if __name__ == "__main__":
    # Sanity test & shape overview
    model = SimpleResidualUViT(height=64, width=64, channels_of_stage=[16,32,64],
                               num_res_down_blocks=[2,3,4], num_res_up_blocks=[6,4,3])
    print(model)
    x = torch.randn(1, 16, 64, 64)
    print("Input:", x.shape)
    h = model.initial_emb(x)
    print("After initial_emb:", h.shape)
    pos_skips, neg_skips = [], []
    # Down
    for i, (blocks, down) in enumerate(zip(model.res_down, model.down_ops)):
        for j, blk in enumerate(blocks):
            h = blk(h)
            print(f"ResDown {i}-{j}:", h.shape)
        pos_skips.append(h)
        h = down(h)
        print(f"After down {i}:", h.shape)
        neg_skips.append(h)
    # Middle
    h = h + model.pos_emb
    print("After pos_emb:", h.shape)
    for k, t in enumerate(model.transformers):
        h = h + t(h)
        print(f"After transformer {k}:", h.shape)
    # Up
    for i, (up, blocks) in enumerate(zip(model.up_ops, model.res_up)):
        neg = neg_skips.pop()
        h = h - neg
        print(f"After subtract neg_skip {i}:", h.shape)
        h = up(h)
        print(f"After up {i}:", h.shape)
        pos = pos_skips.pop()
        h = h + pos
        print(f"After add pos_skip {i}:", h.shape)
        for j, blk in enumerate(blocks):
            h = blk(h)
            print(f"ResUp {i}-{j}:", h.shape)
    # Final
    h = model.norm_out(h)
    print("After norm_out:", h.shape)
    h = model.act_out(h)
    print("After act_out:", h.shape)
    y = model.conv_out(h)
    print("Output:", y.shape)
