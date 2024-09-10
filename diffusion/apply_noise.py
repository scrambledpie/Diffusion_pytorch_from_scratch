import torch

from diffusion.noise_schedule import cosine_schedule


def apply_noise(x_input: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Make the inputs ad targets for UNet
    loss = MSE( Unet(x_noisy, noise_sd***2) - x_noise )
    Args:
        x_input: clean inpt

    RETURNS
        x_noisy: x_input * signal_sd + x_noise * noise_sd
        noise_sd: random time during diffusion
        x_noise: tesnro the same shape as x, the target of regression
    """
    device = x_input.device
    x_noise = torch.normal(mean=torch.zeros_like(x_input)).to(device)

    # get mixture ratios
    diffusion_times = torch.rand((x_input.shape[0], 1, 1, 1)).to(device)
    noise_sd, signal_sd = cosine_schedule(diffusion_times)
    x_noisy = x_input * signal_sd + x_noise * noise_sd

    return x_noisy, noise_sd, x_noise
