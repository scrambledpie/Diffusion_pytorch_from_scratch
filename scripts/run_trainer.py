import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.apply_noise import apply_noise
from diffusion.unet import UNet
from dataset.datasets import Flowers, CelebA, CelebA10k

from folders import make_new_folders


CHECKPOINT_DIR, LOG_DIR = make_new_folders()


def train_model(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataloader: DataLoader,
    epochs:int = 10,
):
    model.train()

    epoch_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        for i, x_input in enumerate(dataloader):
            batch_start = time.time()

            # clean images (batch, channels, height, width)
            x_input = x_input.to(model.device)

            # add noise to the images
            x_diffused, x_noise, noise_sd = apply_noise(x_input)

            # forward
            optimizer.zero_grad()
            x_noise_predicted = model(x_diffused, noise_sd**2)
            mse_loss = F.mse_loss(x_noise_predicted, x_noise)

            # backward
            mse_loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            print(f"{epoch}. {i}: {mse_loss:.3f} {batch_time:.3f} seconds")

        # Track metrics, save checkpoint
        epoch_time = time.time() - epoch_start
        checkpoint_file = CHECKPOINT_DIR / f"{epoch} epoch.pt"
        torch.save(model.state_dict(), f=checkpoint_file)
        print(
            f"{epoch}: epoch time: {epoch_time:.4} seconds\n"
            f"Saved model: {checkpoint_file}\n"
        )


def main():
    dataset = CelebA10k()
    dataloader = DataLoader(
        dataset,
        batch_size=300,
        pin_memory=True,
        shuffle=True,
    )

    model = UNet(device="cuda")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        epochs=10000,
    )


if __name__=="__main__":
    main()
