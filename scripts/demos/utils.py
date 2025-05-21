import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def save_images_grid(
    images: list[torch.Tensor | np.ndarray],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = axes.flatten() if n_images > 1 else [axes]

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    # remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        # close the figure
    else:
        plt.show(block=True)  # 主要修改点在这里
