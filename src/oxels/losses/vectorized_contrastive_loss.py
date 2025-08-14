import torch


def vectorized_contrastive_loss(oview1, oview2, indices, mask1, mask2, eps=1e-8, norm="inf") -> torch.Tensor:
    """
    Vectorized version of contrastive_loss.

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
    B, C, H, W = oview1.shape
    N = H * W

    # flatten spatial dims
    o1 = oview1.view(B, C, N)  # (B, C, N)
    o2 = oview2.view(B, C, N)

    # gather o2 and mask2 at each batch’s indices
    idx = indices.long()
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, N)
    o2g = torch.gather(o2, dim=2, index=idx_exp)  # (B, C, N)
    m2g = torch.gather(mask2.float(), dim=1, index=idx)  # (B, N)

    m1 = mask1.float()  # (B, N)

    # pairwise differences: broadcast over batch dims
    # diff[b1,b2] = o1[b1] - o2g[b2]
    diff = o1.unsqueeze(1) - o2g.unsqueeze(0)  # (B, B, C, N)

    # max‐over‐channels & scale
    if norm == "inf":
        diff = 0.5 * diff.abs().max(dim=2).values  # (B, B, N)
    elif norm == "l1":
        diff = 0.5 * diff.abs().mean(dim=2)
    elif norm == "l2":
        diff = 0.5 * diff.square().mean(dim=2)
    else:
        raise NotImplementedError("norm must be in ['inf', 'l1', 'l2']")

    # per‐pair loss elements
    loss_elem = 0.25 - diff + diff**2  # (B, B, N)

    # pairwise combined masks
    mask_pair = m1.unsqueeze(1) * m2g.unsqueeze(0)  # (B, B, N)

    # we only want b1 < b2
    pair_selector = torch.triu(torch.ones(B, B, device=oview1.device), diagonal=1)
    pair_selector = pair_selector.unsqueeze(-1)  # (B, B, 1)

    # compute masked mean per pair
    num = (loss_elem * mask_pair * pair_selector).sum(dim=2)  # (B, B)
    den = (mask_pair * pair_selector).sum(dim=2).clamp(min=eps)  # (B, B)
    per_pair_loss = num / den  # (B, B)

    # average over all b1<b2 pairs
    total_pairs = pair_selector.sum()
    return per_pair_loss.sum() / total_pairs
