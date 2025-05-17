import torch


def vectorized_loss(oview1, oview2, indices, flags, mask1, mask2) -> torch.Tensor:
    """
    Vectorized version of the original loss function for improved performance.

    Computes a custom contrastive-style loss between two spatially transformed views
    of an image representation (oxel space), using a soft matching scheme and visibility masks.

    Parameters
    ----------
    oview1 : torch.Tensor, shape (B, C, H, W)
        First view in oxel space (e.g., output of a model from image 1).
    oview2 : torch.Tensor, shape (B, C, H, W)
        Second view in oxel space (e.g., output from the transformed version).
    indices : torch.Tensor, shape (B, H*W)
        Flattened indices mapping each pixel in `oview1` to its corresponding position in `oview2`.
        Should be integers in the range [0, H*W).
    flags : torch.Tensor, shape (B, H*W)
        Binary tensor indicating which matches between `oview1` and `oview2` are correct (1) or incorrect (0).
    mask1 : torch.Tensor, shape (B, H*W)
        Binary mask indicating which positions in `oview1` are valid and should be used for loss computation.
    mask2 : torch.Tensor, shape (B, H*W)
        Binary mask indicating which positions in `oview2` are valid for loss computation.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the averaged loss over all valid batches.
    """
    B, C, H, W = oview1.shape
    N = H * W

    # flatten spatial dims
    o1 = oview1.view(B, C, N)
    o2 = oview2.view(B, C, N)

    # ensure correct dtype
    indices = indices.long()
    flags = flags.float()
    m1 = mask1.float()
    m2 = mask2.float()

    # gather oview2 at the mapped positions
    # idx_exp: (B, C, N)
    idx_exp = indices.unsqueeze(1).expand(-1, C, -1)
    o2g = torch.gather(o2, dim=2, index=idx_exp)

    # per‐element difference and max‐over‐channels
    diff = o1 - o2g  # (B, C, N)
    max_diff = 0.5 * diff.abs().max(dim=1).values  # (B, N)

    # per‐sample average of flags
    flags_mean = flags.mean(dim=1, keepdim=True)  # (B, 1)

    # base loss term (same for all N)
    base = 0.25 * (1.0 - flags_mean)  # (B, 1)

    # per‐element loss
    loss_elem = base + (max_diff * (2 * flags - 1) + max_diff**2)  # (B, N)

    # build combined mask: mask1[b,n] * mask2[b, indices[b,n]]
    m2g = torch.gather(m2, dim=1, index=indices)  # (B, N)
    mask = m1 * m2g  # (B, N)

    # per‐sample reduction
    # sum(loss * mask) / sum(mask)
    loss_per_sample = (loss_elem * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (B,)

    # final scalar
    return loss_per_sample.mean()
