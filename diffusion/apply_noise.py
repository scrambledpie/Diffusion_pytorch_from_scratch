import torch

from diffusion.noise_schedule import cosine_schedule, linear_schedule


def apply_noise(
    x_input: torch.Tensor,
    times: list[float]|None=None,
    ) -> tuple[torch.Tensor]:
    """
    Make the inputs ad targets for UNet
    loss = MSE( Unet(x_diffused, noise_sd***2) - x_noise )

    Args:
        x_input: clean input images (batch. C, H, W)
        times: batch list of diffusion times, 0 < times[i] < 1, if None, random
               times are generated.

    RETURNS
        x_diffused: (x_input * signal_sd) + (x_noise * noise_sd)
        x_noise: tensor the same shape as x, the target of regression
        noise_sd: (batch, 1, 1, 1) tensor of noise standard deviations
    """
    device = x_input.device
    x_noise = torch.normal(mean=torch.zeros_like(x_input)).to(device)


    # get mixture ratios
    times_shape = (x_input.shape[0], 1, 1, 1)
    if times is None:
        diffusion_times = torch.rand(times_shape).to(device)
    else:
        diffusion_times = torch.tensor(times).reshape(times_shape).to(device)

    signal_sd, noise_sd  = cosine_schedule(diffusion_times)
    x_diffused = x_input * signal_sd + x_noise * noise_sd

    return x_diffused, x_noise, noise_sd
