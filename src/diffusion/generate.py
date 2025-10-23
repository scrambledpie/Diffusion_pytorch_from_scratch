import numpy as np
import torch

from diffusion.noise_schedule import cosine_schedule
from diffusion.unet import UNet


def generate_images(
    *,
    model: UNet,
    num_steps:int=3,
    batchsize:int=25,
    height:int=109,
    width:int=89,
    seed:int=432
) -> torch.tensor:
    """
    Generate some nice new images from scratch. Use the given UNet model and
    start with poure noise and over num_steps jump back and forth to the
    full predicted image.

    Returns
    -------
    x_t_series : torch.tensor
        (batch_size, num_steps, C, H, W)
    """
    model.train(False)

    device = model.device
    times = torch.linspace(1, 0, num_steps).to(device)

    # images a time=1, full noise
    np.random.seed(seed)
    x_np = np.random.normal(size=(batchsize, 3, height, width))
    x_t = torch.Tensor(x_np).to(device)

    ones_tensor = torch.ones((batchsize, 1, 1, 1)).to(device)
    x_t_series = [x_t.detach().clone()]

    for i in range(num_steps - 1):
        # from x_t predict current noise and signal
        time_tensor_t = ones_tensor * times[i]
        signal_sd_t, noise_sd_t = cosine_schedule(time_tensor_t)

        x_noise_pred_t = model(x_t, noise_sd_t**2)
        x_signal_pred_t = (x_t - noise_sd_t * x_noise_pred_t) / signal_sd_t

        # compose next noise and signal
        time_tensor_t1 = ones_tensor * times[i + 1]
        signal_sd_t1, noise_sd_t1 = cosine_schedule(time_tensor_t1)
        x_t1 = x_signal_pred_t * signal_sd_t1 + x_noise_pred_t * noise_sd_t1

        # get ready for the next step
        x_t = x_t1
        x_t_series.append(x_signal_pred_t.detach().clone())

    x_t_series = torch.stack(x_t_series, dim=1)

    return x_t_series

