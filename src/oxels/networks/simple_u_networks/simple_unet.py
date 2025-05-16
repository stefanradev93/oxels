from collections.abc import Sequence

import torch
import torch.nn as nn

from .simple_downsample import SimpleDownSample
from .simple_upsample import SimpleUpSample
from .simple_residual_block import SimpleResidualBlock
from .simple_self_attention import SimpleAttention
from .simple_norm import SimpleNorm


class SimpleUNet(nn.Module):
    """
    Simple UNet without time embeddings, using SimpleNorm.

    Architecture:
      - Initial 3×3 conv to bring input to channels_of_stage[0]
      - Down path: ResBlock(s) (no skip) + optional attention + downsample, collecting skips after each ResBlock
      - Middle: ResBlock + attention + ResBlock
      - Up path: for each level: for i in 0..num_res_blocks[level]: ResBlock(has_skip=(i>0)) + optional attention, then upsample
        skipping uses collected skips
      - Final SimpleNorm -> SiLU -> 3×3 Conv2d(zero-init)
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels_of_stage: Sequence[int],
        in_channels: int = None,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] = None,
        residual_dropout: Sequence[float] = None,
        has_attention: Sequence[bool] = None,
        num_heads: int = 4,
        norm_groups: int = 8,
    ):
        super().__init__()
        L = len(channels_of_stage)
        if in_channels is None:
            in_channels = channels_of_stage[0]
        if num_res_blocks is None:
            num_res_blocks = [1] * L
        if has_attention is None:
            has_attention = [False] * L

        # Initial conv
        self.initial_conv = nn.Conv2d(in_channels, channels_of_stage[0], kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.initial_conv.weight, gain=1.0)
        nn.init.zeros_(self.initial_conv.bias)

        # Down path
        self.res_down = nn.ModuleList()
        self.attn_down = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        for i in range(L):
            # Res blocks
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks[i]):
                blocks.append(
                    SimpleResidualBlock(
                        out_channels=channels_of_stage[i],
                        activation=nn.SiLU,
                        has_skip=False,
                        dropout=residual_dropout[i],
                    )
                )
                attns.append(
                    SimpleAttention(num_heads=num_heads, channel_dim=channels_of_stage[i])
                    if has_attention[i]
                    else nn.Identity()
                )
            self.res_down.append(blocks)
            self.attn_down.append(attns)
            # Downsample
            out_ch = channels_of_stage[i + 1] if i < L - 1 else channels_of_stage[-1]
            self.down_ops.append(SimpleDownSample(channels_of_stage[i], out_ch))

        # Middle: one Res, one attention, one Res
        self.mid_res1 = SimpleResidualBlock(
            out_channels=channels_of_stage[-1], activation=nn.SiLU, has_skip=False, dropout=residual_dropout[-1]
        )
        self.mid_attn = SimpleAttention(num_heads=num_heads, channel_dim=channels_of_stage[-1])
        self.mid_res2 = SimpleResidualBlock(
            out_channels=channels_of_stage[-1], activation=nn.SiLU, has_skip=False, dropout=residual_dropout[-1]
        )

        # Up path
        self.res_up = nn.ModuleList()
        self.attn_up = nn.ModuleList()
        self.up_ops = nn.ModuleList()
        for i in reversed(range(L)):
            # Res blocks plus one for skip
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for j in range(num_res_blocks[i] + 1):
                blocks.append(
                    SimpleResidualBlock(
                        out_channels=channels_of_stage[i],
                        activation=nn.SiLU,
                        has_skip=(j < num_res_blocks[i]),
                        dropout=residual_dropout[i],
                    )
                )
                attns.append(
                    SimpleAttention(num_heads=num_heads, channel_dim=channels_of_stage[i])
                    if has_attention[i]
                    else nn.Identity()
                )
            self.res_up.append(blocks)
            self.attn_up.append(attns)
            # Upsample
            in_ch = channels_of_stage[i + 1] if i < L - 1 else channels_of_stage[-1]
            self.up_ops.append(SimpleUpSample(in_ch, channels_of_stage[i]))

        # Final projection
        self.norm_out = SimpleNorm(
            channel_dim=channels_of_stage[0],
            method="group",
            groups=min(norm_groups, channels_of_stage[0]),
            center=True,
            scale=True,
        )
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(channels_of_stage[0], out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.initial_conv(x)
        skips = []
        # Down path
        for i in range(len(self.res_down)):
            blocks = self.res_down[i]
            attns = self.attn_down[i]
            for blk, att in zip(blocks, attns):
                h = blk(h)
                h = att(h)
                skips.append(h)
            h = self.down_ops[i](h)
        # Middle
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        # Up path
        # skips = list(reversed(skips))
        for i in range(len(self.res_up)):
            blocks = self.res_up[i]
            attns = self.attn_up[i]
            h = self.up_ops[i](h)
            for j, (blk, att) in enumerate(zip(blocks, attns)):
                skip_h = (
                    skips.pop() if j < len(blocks) - 1 else None
                )  # UpBlocks are one more and last one doesnt have skip.
                h = blk(h, skip_h=skip_h)
                h = att(h)
        # Final
        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)


if __name__ == "__main__":
    # Sanity test
    from tqdm import tqdm
    import torch.nn.functional as F

    # Sanity test
    b_ch = 16
    simple_diffusion_512 = {
        "height": 256,
        "width": 256,
        "in_channels": 3,
        "out_channels": b_ch,
        "channels_of_stage": [1 * b_ch, 2 * b_ch, 4 * b_ch, 4 * b_ch, 8 * b_ch, 8 * b_ch],
        "num_res_blocks": [1, 2, 4, 6, 8, 8],
        "residual_dropout": [0.1] * 6,
        "has_attention": [False, False, False, False, True, True],
        "num_heads": 4,
        "norm_groups": 8,
    }
    config = simple_diffusion_512
    model = SimpleUNet(
        height=config["height"],
        width=config["width"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channels_of_stage=config["channels_of_stage"],
        num_res_blocks=config["num_res_blocks"],
        residual_dropout=config["residual_dropout"],
        has_attention=config["has_attention"],
        num_heads=config["num_heads"],
        norm_groups=config["norm_groups"],
    )
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 24
    for i in tqdm(range(100)):
        optimizer.zero_grad()
        x = torch.randn(bs, config["in_channels"], config["height"], config["width"]).to(device)
        y = model(x)
        loss = F.mse_loss(y, torch.randn(bs, config["out_channels"], config["height"], config["width"]).to(device))
        loss.backward()
        optimizer.step()
    print("Training complete.")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    # down
    h = model.initial_conv(x)
    skips = []
    for i, (blocks, atts, down) in enumerate(zip(model.res_down, model.attn_down, model.down_ops)):
        for j, (blk, att) in enumerate(zip(blocks, atts)):
            h = blk(h)
            h = att(h)
            print(f"Down level {i} block {j}:", h.shape)
            skips.append(h)
        h = down(h)
        print(f"After down {i}:", h.shape)
    # middle
    h = model.mid_res1(h)
    print("After mid_res1:", h.shape)
    h = model.mid_attn(h)
    print("After mid_attn:", h.shape)
    h = model.mid_res2(h)
    print("After mid_res2:", h.shape)
    # up
    for i, (up, blocks, atts) in enumerate(zip(model.up_ops, model.res_up, model.attn_up)):
        h = up(h)
        print(f"After up {i}:", h.shape)
        for j, (blk, att) in enumerate(zip(blocks, atts)):
            skip_h = (
                skips.pop() if j < len(blocks) - 1 else None
            )  # UpBlocks are one more and last one doesnt have skip.
            h = blk(h, skip_h=skip_h)
            h = att(h)
            print(f"Up level {i} block {j}:", h.shape)
    # final
    h = model.norm_out(h)
    print("After norm_out:", h.shape)
    h = model.act_out(h)
    print("After act_out:", h.shape)
    y = model.conv_out(h)
    print("Output:", y.shape)
