from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_images(
    x: torch.Tensor,
    filename:Path,
    ncol:int=5,
    nrow:int=5,
) -> None:
    """
    Given a tensor of generated images x: (batch, channels, height, width)
    make a nice picture and save a jpg.
    """
    # tensor(batch, 3, 64, 64) -> np.array(batch, 64, 64, 3)
    x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)

    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    ax = ax.reshape(-1)

    for ax_i in ax:
        ax_i.set_axis_off()

    for ax_i, x_i in zip(ax, x):
        ax_i.imshow(x_i)

    fig.tight_layout()
    fig.savefig(filename)
    print(f"Saved {filename}")
