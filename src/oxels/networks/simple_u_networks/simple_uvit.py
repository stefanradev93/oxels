import torch
import torch.nn as nn

from .simple_downsample import SimpleDownSample
from .simple_upsample import SimpleUpSample
from .simple_residual_block import SimpleResidualBlock
from .simple_transformer_block import SimpleTransformerBlock
from .simple_norm import SimpleNorm


class SimpleUViT(nn.Module):
    DEFAULT_NUM_BLOCKS_PER_STAGE = 3
    """
    Plain U-ViT without time embeddings, using SimpleNorm for normalization.

    Architecture:
      - Initial 3×3 conv
      - Down path: sequences of ResBlocks (no internal skips) followed by downsampling, collecting skips
      - Middle: add spatial positional bias + Transformer blocks (residual)
      - Up path: upsampling followed by ResBlocks using collected skips
      - Final SimpleNorm -> SiLU -> 3×3 Conv2d (zero init)
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels_of_stage: list,
        in_channels: int = None,
        out_channels: int = 1,
        num_res_blocks: list = None,
        residual_dropout: list = None,
        num_transformer_blocks: int = 4,
        num_heads: int = 4,
        transformer_expansion: int = 4,
        transformer_dropout: float = 0.0,
        norm_groups: int = 8,
    ):
        super().__init__()
        L = len(channels_of_stage)
        if in_channels is None:
            in_channels = channels_of_stage[0]
        if num_res_blocks is None:
            num_res_blocks = [3] * L
        if residual_dropout is None:
            residual_dropout = [0.0] * L

        # Initial conv: 3×3
        self.initial_emb = nn.Conv2d(in_channels, channels_of_stage[0], kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.initial_emb.weight, gain=1.0)
        nn.init.zeros_(self.initial_emb.bias)

        # Down path
        self.res_down = nn.ModuleList()
        self.down_ops = nn.ModuleList()
        for i in range(L):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks[i]):
                blocks.append(
                    SimpleResidualBlock(
                        out_channels=channels_of_stage[i],
                        activation=nn.SiLU,
                        has_skip=False,
                        dropout=residual_dropout[i],
                    )
                )
            self.res_down.append(blocks)
            out_ch = channels_of_stage[i + 1] if i < L - 1 else channels_of_stage[-1]
            self.down_ops.append(SimpleDownSample(channels_of_stage[i], out_ch))

        # Positional embedding for middle
        mid_h = height // (2**L)
        mid_w = width // (2**L)
        self.pos_emb = nn.Parameter(torch.randn(1, channels_of_stage[-1], mid_h, mid_w) * 0.01)

        # Transformer blocks (residual connections inside)
        self.transformers = nn.ModuleList(
            [
                SimpleTransformerBlock(
                    channel_dim=channels_of_stage[-1],
                    expansion=transformer_expansion,
                    num_heads=num_heads,
                    dropout=transformer_dropout,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # Up path
        self.up_ops = nn.ModuleList()
        self.res_up = nn.ModuleList()
        for i in reversed(range(L)):
            in_ch = channels_of_stage[i + 1] if i < L - 1 else channels_of_stage[-1]
            self.up_ops.append(SimpleUpSample(in_ch, channels_of_stage[i]))
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks[i]):
                blocks.append(
                    SimpleResidualBlock(
                        out_channels=channels_of_stage[i],
                        activation=nn.SiLU,
                        has_skip=True,
                        dropout=residual_dropout[i],
                    )
                )
            self.res_up.append(blocks)

        # Final projection: SimpleNorm -> SiLU -> zero-init Conv2d
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
        h = self.initial_emb(x)
        skips = []
        # Down path
        for blocks, down in zip(self.res_down, self.down_ops):
            for blk in blocks:
                h = blk(h)
                skips.append(h)
            h = down(h)

        # Middle
        h = h + self.pos_emb
        for t in self.transformers:
            h = t(h)

        # Up path
        for up, blocks in zip(self.up_ops, self.res_up):
            h = up(h)
            for blk in blocks:
                h = blk(h, skip_h=skips.pop())

        # Final projection
        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)


if __name__ == "__main__":
    from tqdm import tqdm
    import torch.nn.functional as F

    # Sanity test
    b_ch = 64
    simple_diffusion_512 = {
        "height": 512,
        "width": 512,
        "in_channels": 3,
        "out_channels": 16,
        "channels_of_stage": [1 * b_ch, 2 * b_ch, 4 * b_ch],
        "num_res_blocks": [2, 2, 2],
        "residual_dropout": [0.1, 0.1, 0.1],
        "num_transformer_blocks": 24,
        "num_heads": 4,
        "transformer_expansion": 4,
        "transformer_dropout": 0.2,
        "norm_groups": 8,
    }
    config = simple_diffusion_512
    model = SimpleUViT(
        height=config["height"],
        width=config["width"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channels_of_stage=config["channels_of_stage"],
        num_res_blocks=config["num_res_blocks"],
        residual_dropout=config["residual_dropout"],
        num_transformer_blocks=config["num_transformer_blocks"],
        num_heads=config["num_heads"],
        transformer_expansion=config["transformer_expansion"],
        transformer_dropout=config["transformer_dropout"],
        norm_groups=config["norm_groups"],
    )
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 1
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
    # Down path
    h = model.initial_emb(x)
    print("After initial conv:", h.shape)
    skips = []
    for i, (blocks, down) in enumerate(zip(model.res_down, model.down_ops)):
        for j, blk in enumerate(blocks):
            h = blk(h)
            print(f"Down block {i}-{j} output:", h.shape)
            skips.append(h)
        h = down(h)
        print(f"After downsample {i}:", h.shape)
    # Middle
    h = h + model.pos_emb
    print("After pos_emb:", h.shape)
    for k, t in enumerate(model.transformers):
        h = h + t(h)
        print(f"After transformer {k}:", h.shape)
    # Up path
    skips = list(reversed(skips))
    idx = 0
    for i, (up, blocks) in enumerate(zip(model.up_ops, model.res_up)):
        h = up(h)
        print(f"After upsample {i}:", h.shape)
        for j, blk in enumerate(blocks):
            skip_h = skips[idx]
            h = blk(h, skip_h=skip_h)
            print(f"Up block {i}-{j} output:", h.shape)
            idx += 1
    # Final
    h = model.norm_out(h)
    print("After final norm:", h.shape)
    h = model.act_out(h)
    print("After final activation:", h.shape)
    y = model.conv_out(h)
    print("Final output shape:", y.shape)
