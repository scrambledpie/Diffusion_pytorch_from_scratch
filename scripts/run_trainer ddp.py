import datetime
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.apply_noise import apply_noise
from diffusion.unet import UNet
from dataset.datasets import Flowers, CelebA, CelebA10k

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

ROOT_DIR = Path(__file__).parent

CHECKPOINTS_ROOT = ROOT_DIR / "checkpoints"
num_folders = len(list(CHECKPOINTS_ROOT.glob("*")))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"{num_folders}_{timestamp}"
CHECKPOINT_DIR = CHECKPOINTS_ROOT / MODEL_NAME
CHECKPOINT_DIR.mkdir()

LOG_DIR = ROOT_DIR / "tensorboard_logs" / MODEL_NAME
writer = SummaryWriter(log_dir=LOG_DIR)


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataloader: DataLoader,
    use_ddp:bool,
    epochs:int = 10,
    rank:int=0,
):
    model.train()
    epoch_start = time.time()

    model = model.to(rank)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    iter_count = 0
    epoch_loss = 0
    epoch_iters = 0

    for epoch in range(epochs):
        epoch_time = time.time() - epoch_start

        if rank == 0 and epoch > 0:
            checkpoint_file = CHECKPOINT_DIR / f"{epoch} epoch.pt"
            if use_ddp:
                torch.save(model.module.state_dict(), f=checkpoint_file)
            else:
                torch.save(model.state_dict(), f=checkpoint_file)

            print(
                f"{epoch - 1}: epoch time: {epoch_time:.4} seconds\n"
                f"Saved model: {checkpoint_file}\n"
            )
            writer.add_scalar("epoch time", epoch_time, epoch)
            writer.add_scalar("epoch loss", epoch_loss / epoch_iters, epoch)

        epoch_start = time.time()
        epoch_loss = 0
        epoch_iters = 0
        for i, x_input in enumerate(dataloader):
            batch_start = time.time()

            x_input = x_input.to(rank)

            # clean image and pure noise
            x_diffused, x_noise, noise_sd = apply_noise(x_input)

            # forward
            optimizer.zero_grad()
            x_noise_predicted = model(x_diffused, noise_sd**2)
            mse_loss = F.mse_loss(x_noise_predicted, x_noise)

            # backward
            mse_loss.backward()
            optimizer.step()

            epoch_loss += mse_loss
            epoch_iters = 0

            batch_time = time.time() - batch_start
            if rank == 0:
                print(f"{epoch}. {i}: {mse_loss:.3f} {batch_time:.3f} seconds")
                writer.add_scalar("mse_loss", mse_loss, iter_count)
                iter_count += 1


def main(rank:int, worldsize:int=2, use_ddp:bool=False):
    if use_ddp:
        ddp_setup(rank, worldsize)

    dataset = Flowers()
    dataloader = DataLoader(
        dataset,
        batch_size=300,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) if use_ddp else None
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
        use_ddp=use_ddp,
    )
    if use_ddp:
        destroy_process_group()


if __name__=="__main__":
    if 1==11:
        world_size = torch.cuda.device_count()
        print(f"Starting {world_size} GPUs")
        mp.spawn(main, args=(world_size, True), nprocs=world_size)
    else:
        main(rank=0, use_ddp=False)
