import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


from diffusion.unet.unet import UNet
from dataset.datasets import Flowers, CelebA, CelebA10k
from diffusion.train import train_model

from folders import make_new_folders




def main():
    checkpoint_dir, log_dir = make_new_folders()

    dataset = CelebA10k()
    dataloader = DataLoader(
        dataset,
        batch_size=300,
        pin_memory=True,
        shuffle=True,
    )

    model = UNet(device="cuda")
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        checkpoint_dir=checkpoint_dir,
        epochs=3,
        tensorboard_writer=SummaryWriter(logdir=log_dir)
    )


if __name__=="__main__":
    main()
