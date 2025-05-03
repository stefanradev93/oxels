import matplotlib.pyplot as plt
import numpy as np


def plot_channels(im: np.ndarray, cmap: str = 'gray'):
    """
    Visualizes a multi-channel image (shape: [C, H, W]) using subplots.

    Args:
        image (np.ndarray or torch.Tensor): Input image of shape (C, H, W)
        cmap (str): Color map for plotting, e.g., 'gray' or 'viridis'
    """
    if hasattr(im, 'detach'):
        im = im.detach().cpu().numpy()

    if im.ndim != 3:
        raise ValueError("Input image must have shape (C, H, W)")

    channels, height, width = im.shape
    cols = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(channels):
        axes[i].imshow(im[i], cmap=cmap)
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    # Hide unused axes
    for i in range(channels, len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    return fig
