import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.noise_schedule import apply_noise
from diffusion.unet.unet import UNet
from dataset.datasets import Flowers, CelebA, CelebA10k

from folders import make_new_folders

from tensorboardX import SummaryWriter

# DDP Boiler Plate Start
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
# DDP Boiler Plate End


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataloader: DataLoader,
    epochs:int = 10,
    rank:int=0,
):
    if rank == 0:
        CHECKPOINT_DIR, LOG_DIR = make_new_folders()
        writer = SummaryWriter(log_dir=LOG_DIR)

    model.train()
    epoch_start = time.time()

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    iter_count = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_iters = 0
        for i, x_input in enumerate(dataloader):
            batch_start = time.time()

            # clean image (batch, channels, height, width)
            x_input = x_input.to(rank)

            # noisy image, only noise, and noise standard devitaions
            x_diffused, x_noise, noise_sd = apply_noise(x_input)

            # forward
            optimizer.zero_grad()
            x_noise_predicted = model(x_diffused, noise_sd**2)
            mse_loss = F.mse_loss(x_noise_predicted, x_noise)

            # backward
            mse_loss.backward()
            optimizer.step()

            if rank == 0:
                epoch_loss += mse_loss
                epoch_iters += 1
                batch_time = time.time() - batch_start
                print(f"{epoch}. {i}: {mse_loss:.3f} {batch_time:.3f} seconds")
                writer.add_scalar("mse_loss", mse_loss, iter_count)
                iter_count += 1

        if rank == 0:
            epoch_time = time.time() - epoch_start
            checkpoint_file = CHECKPOINT_DIR / f"{epoch} epoch.pt"
            torch.save(model.module.state_dict(), f=checkpoint_file)

            print(
                f"{epoch}: epoch time: {epoch_time:.4} seconds\n"
                f"Saved model: {checkpoint_file}\n"
            )
            writer.add_scalar("epoch time", epoch_time, epoch)
            writer.add_scalar("epoch loss", epoch_loss / epoch_iters, epoch)


def main(rank:int, world_size:int=2):
    ddp_setup(rank=rank, world_size=world_size)
    dataset = CelebA10k()
    dataloader = DataLoader(
        dataset,
        batch_size=300,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    model = UNet()

    print(f"Rank: {rank}")

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=20000,
        rank=rank,
    )

    destroy_process_group()


if __name__=="__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting {world_size} GPUs")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
