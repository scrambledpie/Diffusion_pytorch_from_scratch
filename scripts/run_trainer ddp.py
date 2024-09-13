import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from diffusion.unet.unet import UNet
from dataset.datasets import Flowers, CelebA, CelebA10k
from diffusion.train import train_model

from folders import make_new_folders

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
    model.train()
    model = DDP(model, device_ids=[rank])

    print(f"Rank: {rank}")

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # training setting for all nodes
    train_model_kwargs = dict(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=20000,
        verbose=False,
    )

    if rank == 0:
        # extra logging settings just for 0th node
        checkpoint_dir, log_dir = make_new_folders()
        train_model_kwargs.update(
            dict(
                checkpoint_dir=checkpoint_dir,
                verbose = True,
                tensorboard_writer=SummaryWriter(logdir=log_dir),
            )
        )

    train_model(**train_model_kwargs)
    destroy_process_group()


if __name__=="__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting {world_size} GPUs")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
