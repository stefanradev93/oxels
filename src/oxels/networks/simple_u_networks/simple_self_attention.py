import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_norm import SimpleNorm


class SimpleAttention(nn.Module):
    """
    Multi-head self-attention for (B, C, H, W) inputs, using SimpleNorm.

    Adheres to the pseudocode:

      def self_attention(x, text_emb):
          B, HW, C = x.shape
          head_dim = C // num_heads
          x_norm = Normalize(x)
          q = DenseGeneral(x_norm, (num_heads, head_dim))
          k = DenseGeneral(x_norm, (num_heads, head_dim))
          v = DenseGeneral(x_norm, (num_heads, head_dim))
          q = NormalizeWithBias(q)
          k = NormalizeWithBias(k)
          q = q * head_dim**-0.5
          weights = einsum("bqhd,bkhd->bhqk", q, k)
          weights = softmax(weights)
          attn_vals = einsum("bhqk,bkhd->bqhd", weights, v)
          out = DenseGeneral(attn_vals, C, kernel_init=zeros)
          return out

    Steps implemented here:
      1) Flatten to (B, HW, C) and apply input norm
      2) Linear projections for Q, K, V
      3) Reshape to (B, HW, num_heads, head_dim)
      4) Normalize Q and K with SimpleNorm (center=True)
      5) Scale Q by head_dim^-0.5
      6) Compute attention weights and apply softmax
      7) Compute attended values
      8) Merge heads and final zero-init projection back to C
      9) Reshape to (B, C, H, W)
    No residual connection in this block.
    """

    def __init__(self, num_heads: int, channel_dim: int):
        super(SimpleAttention, self).__init__()
        assert channel_dim % num_heads == 0, "channel_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.c_dim = channel_dim
        self.head_dim = channel_dim // num_heads

        # 1) Normalize input x (no bias shift)
        self.input_norm = SimpleNorm(channel_dim, method="layer", center=False, scale=True)

        # 2) Linear projections for Q, K, V to C total dims
        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)

        # Initialize Q/K/V with Xavier uniform
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            nn.init.zeros_(proj.bias)

        # 4) Normalize Q and K (layer norm over head dim)
        self.q_norm = SimpleNorm(self.head_dim, method="layer", center=True, scale=True)
        self.k_norm = SimpleNorm(self.head_dim, method="layer", center=True, scale=True)

        # 8) Output projection back to C, zero initialized
        self.out_proj = nn.Linear(channel_dim, channel_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        HW = H * W

        # 1) Flatten spatial dims and apply input norm
        x_flat = x.view(B, C, HW)  # (B, C, HW)
        x_norm = self.input_norm(x_flat)  # (B, C, HW)
        x_norm = x_norm.permute(0, 2, 1)  # (B, HW, C)

        # 2) Project to Q, K, V
        q = self.q_proj(x_norm)  # (B, HW, C)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # 3) Reshape for heads
        q = q.view(B, HW, self.num_heads, self.head_dim)
        k = k.view(B, HW, self.num_heads, self.head_dim)
        v = v.view(B, HW, self.num_heads, self.head_dim)

        # 4) Normalize Q, K
        q_perm = q.permute(0, 3, 1, 2)  # (B, head_dim, HW, num_heads)
        k_perm = k.permute(0, 3, 1, 2)  # (B, head_dim, HW, num_heads)
        q = self.q_norm(q_perm)
        k = self.k_norm(k_perm)
        q = q.permute(0, 2, 3, 1)  # (B, HW, num_heads, head_dim)
        k = k.permute(0, 2, 3, 1)  # (B, HW, num_heads, head_dim)

        # 5) Scale Q
        q = q * (self.head_dim**-0.5)

        # 6) Attention weights
        weights = torch.einsum("bqhd,bkhd->bhqk", q, k)  # (B, heads, Q, K)
        weights = F.softmax(weights, dim=-1)

        # 7) Weighted sum
        attn_vals = torch.einsum("bhqk,bkhd->bqhd", weights, v)  # (B, HW, heads, head_dim)

        # 8) Merge heads and final projection
        attn_vals = attn_vals.reshape(B, HW, self.c_dim)  # (B, HW, C)
        out = self.out_proj(attn_vals)  # (B, HW, C)

        # 9) Reshape back to spatial
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return out


if __name__ == "__main__":
    # Simple test
    x = torch.randn(2, 16, 32, 32)
    attn = SimpleAttention(num_heads=4, channel_dim=16)
    y = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
