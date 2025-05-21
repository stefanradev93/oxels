import torch


def original_loss(oview1, oview2, indices, flags, mask1, mask2) -> torch.Tensor:
    """
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
    indices = indices.long()
    flags = flags.float()
    mask1 = mask1.float()
    mask2 = mask2.float()

    B, C, H, W = oview1.shape
    losses = []

    # TODO: vectorize this
    for b in range(B):
        idx = indices[b]
        _mask = mask1[b] * mask2[b][idx]

        diff = oview1[b].reshape((C, H * W)) - oview2[b].reshape((C, H * W))[:, idx]
        # max difference across channels
        max_diff = 0.5 * diff.abs().max(dim=0)[0]

        loss = 0.25 * (1.0 - flags[b].mean()) + (max_diff * (2 * flags[b] - 1) + max_diff**2)
        losses.append((loss * _mask).mean() / _mask.mean())

    return torch.mean(torch.stack(losses))
