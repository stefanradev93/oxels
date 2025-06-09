import torch


def contrastive_loss(oview1, oview2, indices, mask1, mask2) -> torch.Tensor:
    """
    Can be used to augment our original loss with a contrastive component over “negative examples”
    that compares the dense features of different view pairs b and b′ within a batch.

    Parameters
    ----------
    oview1 : torch.Tensor, shape (B, C, H, W)
        First view in oxel space (e.g., output of a model from image 1).
    oview2 : torch.Tensor, shape (B, C, H, W)
        Second view in oxel space (e.g., output from the transformed version).
    indices : torch.Tensor, shape (B, H*W)
        Flattened indices mapping each pixel in `oview1` to its corresponding position in `oview2`.
        Should be integers in the range [0, H*W).
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
    mask1 = mask1.float()
    mask2 = mask2.float()

    B, C, H, W = oview1.shape
    losses = []

    # TODO: vectorize this
    for b2 in range(B):
        idx2 = indices[b2]
        _mask2 = mask2[b2][idx2]
        _oview2 = oview2[b2].reshape((C, H * W))[:, idx2]

        for b1 in range(b2):
            _mask = mask1[b1] * _mask2

            diff = oview1[b1].reshape((C, H * W)) - _oview2
            # max difference across channels
            max_diff = 0.5 * diff.abs().max(dim=0)[0]

            loss = 0.25 - max_diff + max_diff**2
            losses.append((loss * _mask).mean() / _mask.mean())

    return torch.mean(torch.stack(losses))
