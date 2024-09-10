import datetime
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F

from diffusion.noise_schedule import cosine_schedule
from diffusion.unet import UNet
from dataset.reshape_images import FLOWERS_DATASET

def print_shape(params_list: list[torch.tensor]):
    if isinstance(params_list, list):
        return [print_shape(p) for p in params_list]

    if isinstance(params_list, torch.Tensor):
        return tuple(params_list.shape)


CHECKPOINTS_ROOT = Path(__file__).parent / "checkpoints"
num_folders = len(list(CHECKPOINTS_ROOT.glob("*")))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"{num_folders}_{timestamp}"
CHECKPOINT_DIR = CHECKPOINTS_ROOT / MODEL_NAME
CHECKPOINT_DIR.mkdir()


def train_model(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataset: torch.Tensor,
    batchsize:int =16,
    epochs:int = 10,
    device:str="cuda",
):
    num_batches = dataset.shape[0] // batchsize
    model.train()

    epoch_start = time.time()

    for epoch in range(epochs):
        epoch_time = time.time() - epoch_start

        if epoch > 0:
            checkpoint_file = CHECKPOINT_DIR / f"{epoch} epoch.pt"
            torch.save(model.state_dict(), f=checkpoint_file)
            print(
                f"{epoch -1 }: epoch time: {epoch_time} seconds\n"
                f"Saved model: {checkpoint_file}\n"
            )

        epoch_start = time.time()
        for i in range(num_batches):
            batch_start = time.time()

            # clean image and pure noise
            x_input = dataset[i*batchsize:(i+1)*batchsize, :, :, :].to(device)
            x_noise = torch.normal(mean=torch.zeros_like(x_input)).to(device)

            # get mixture ratios
            diffusion_times = torch.rand((batchsize, 1, 1, 1)).to(device)
            noise_sd, signal_sd = cosine_schedule(diffusion_times)
            x_noisy = x_input * signal_sd + x_noise * noise_sd

            # forward
            optimizer.zero_grad()
            x_noise_predicted = model(x_noisy, noise_sd**2)
            mse_loss = F.mse_loss(x_noise_predicted, x_noise)

            # backward
            mse_loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            print(f"{epoch} {i}: {mse_loss:.3f} {batch_time:.3f} seconds")


def main():

    dataset = FLOWERS_DATASET.to("cuda")

    model = UNet(device="cuda")
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    train_model(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batchsize=800,
    )




main()
