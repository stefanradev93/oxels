def original_loss_func(oview1, oview2, indices, flags, mask1, mask2):
    """
    Parameters
    ----------
    oview1  : shape (B, C, H, W)
        First view in oxel space
    oview2  : (B, C, H, W)
        First view in oxel space
    indices : shape (B, H*W)
        Indices from 0 to H*W-1 of view2.
    flags   : shape (B, H*W)
        Flags to that indicate correct matches between view1 and strasfomred view2:
        flags[i] is true if view2[indices[i]//W, indices[i]%W] should be close to view1[i//w, i%w]
    mask1   : shape (B, H*W)
        Which oxels of view1 should be used for loss computation
    mask2   : shape (B, H*W)
        Which oxels of view2 should be used for loss computation
    """

    indices = indices.long()
    flags = flags.float()
    mask1 = mask1.float()
    mask2 = mask2.float()
    
    B,C,H,W = oview1.shape

    losses = []
    for b in range(B):
        _mask = mask1[b]*(mask2[b][indices[b]])

        diff = oview1[b].reshape((C, H*W)) - oview2[b].reshape((C, H*W))[:,indices[b]]
        max_diff = 0.5 * diff.abs().max(dim=0)[0]

        loss = 0.25*(1.0 - flags.mean()) + (max_diff*(2*flags - 1) + max_diff*max_diff)
    
        losses.append((loss*_mask).mean()/_mask.mean())

    return sum(losses)/B