from pathlib import Path

import time

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from .noise_schedule import apply_noise
from .unet.unet import UNet


def _save_model(
    *,
    model:UNet,
    checkpoint_file:Path,
) -> None:
    """
    Save the model weights to the checkpoint file, handles both normal and
    DDP models.
    """
    if isinstance(
        model,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, f=checkpoint_file)
    print(f"Saved model {checkpoint_file}")


def train_model(
    model: UNet,
    optimizer: torch.optim.AdamW,
    dataloader: DataLoader,
    epochs:int=1,
    checkpoint_dir: Path|None = None,
    verbose:bool=True,
    tensorboard_writer = None,
) -> None:
    """
    Train the UNet for one epoch. All objects are modified in-place, nothing is
    returned.
    """
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()

        for i, x_input in enumerate(dataloader):
            batch_start = time.time()

            # clean images (batch, channels, height, width)
            x_input = x_input.to(model.device)

            # add noise to the images
            # (batch, 3, H, W), (batch, 3, H, W), (batch, 1, 1, 1)
            x_diffused, x_noise, noise_sd = apply_noise(x_input)

            # forward
            optimizer.zero_grad()
            x_noise_predicted = model(x_diffused, noise_sd**2)
            mse_loss = F.mse_loss(x_noise_predicted, x_noise)

            # backward
            mse_loss.backward()
            optimizer.step()

            if verbose:
                # print out batch duration and loss
                batch_time = time.time() - batch_start
                print(f"{epoch}. {i}: {mse_loss:.3f} {batch_time:.3f} seconds")

            if tensorboard_writer is not None:
                # upload metrics to tensorboard
                iters = epoch * len(dataloader) + i
                tensorboard_writer.add_scalar("mse_loss", mse_loss, iters)

        if checkpoint_dir is not None:
            # save the model only if a folder is given
            checkpoint_file = checkpoint_dir / f"{epoch} epoch.pt"
            _save_model(model=model, checkpoint_file=checkpoint_file)

        if verbose:
            epoch_time = time.time() - epoch_start
            print(f"{epoch}: epoch time: {epoch_time:.4} seconds\n")




