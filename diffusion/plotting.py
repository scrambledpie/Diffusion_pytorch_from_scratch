from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_images(
    x: torch.Tensor,
    filename:Path=Path("img.jpg"),
) -> None:
    """
    Given a tensor of generated images x: (batch, channels, height, width)
    make a nice picture and save a jpg.
    """
    # tensor(batch, 3, 64, 64) -> np.array(batch, 64, 64, 3)
    x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)

    fig, ax = plt.subplots(5, 5)
    ax = ax.reshape(-1)

    for ax_i, x_i in zip(ax, x):
        ax_i.imshow(x_i)
        ax_i.set_axis_off()

    fig.tight_layout()
    fig.savefig(filename)
    print(f"Saved {filename}")
