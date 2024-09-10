from pathlib import Path

import numpy as np
import torch

from diffusion.noise_schedule import cosine_schedule
from diffusion.plotting import plot_images
from dataset.reshape_images import FLOWERS_DATASET
from diffusion.unet import UNet


def get_latest_state_dict(xp_id:int=None, epoch:int=None) -> dict:
    """
    Read the checkpoints folder, get the lastest checkpoint from the latest
    experiment.
    """
    checkpoint_dir = Path(__file__).parent / "checkpoints"

    if xp_id is None:
        xp_ids = [int(f.stem.split("_")[0]) for f in checkpoint_dir.glob("*")]
        xp_id = max(xp_ids)

    xp_dir = list(checkpoint_dir.glob(f"{xp_id}_*"))[0]

    if epoch is None:
        epochs = [int(f.stem.split(" ")[0]) for f in xp_dir.glob("*")]
        epoch = max(epochs)

    checkpont_file = list(xp_dir.glob(f"{epoch} *"))[0]
    state_dict = torch.load(checkpont_file, weights_only=True)
    print(f"Loaded model: {checkpont_file}")
    return state_dict


def generate_images(
    *,
    model: UNet,
    num_steps:int=3,
    batchsize:int=25,
    seed:int=42
):
    """
    Generate some nice new images from scratch. Use the given UNet model and
    start with poure noise and over num_steps jump back and forth to the
    full predicted image.
    """

    device = model.device
    times = torch.linspace(1, 0, num_steps).to(device)

    # images a time=1, full noise
    np.random.seed(seed)
    x_np = np.random.normal(size=(batchsize, 3, 64, 64))
    print(x_np.min(), x_np.max())
    x_t = torch.Tensor(x_np).to(device)

    ones_tensor = torch.ones((batchsize, 1, 1, 1)).to(device)

    for i in range(num_steps - 1):
        # from x_t predict current noise and signal
        time_tensor_t = ones_tensor * times[i]
        noise_sd_t, signal_sd_t = cosine_schedule(time_tensor_t)

        x_noise_pred_t = model(x_t, noise_sd_t**2)
        x_signal_pred_t = (x_t - noise_sd_t * x_noise_pred_t) / signal_sd_t

        # compose next noise and signal
        time_tensor_t1 = ones_tensor * times[i + 1]
        noise_sd_t1, signal_sd_t1 = cosine_schedule(time_tensor_t1)
        x_t1 = x_signal_pred_t * signal_sd_t1 + x_noise_pred_t * noise_sd_t1

        # get ready for the next step
        x_t = x_t1

    return x_signal_pred_t



model_state_dict = get_latest_state_dict(xp_id=11, epoch=1903)
model = UNet()
model.load_state_dict(model_state_dict)
x_new = generate_images(model=model, num_steps=40, seed=42)

plot_images(x_new, "img.jpg")
plot_images(FLOWERS_DATASET, "img_training.jpg")
