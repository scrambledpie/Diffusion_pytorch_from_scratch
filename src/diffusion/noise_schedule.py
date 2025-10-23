import torch


def linear_schedule(
    time_steps :list[int],
    beta_0:float = 0.0001,
    beta_1: float = 0.02,
) -> tuple[list[float], list[float]]:
    """
    Create the cumulative product and
    """

    d_beta = (beta_1 - beta_0)
    beta_vals = [1 - (beta_0 + d_beta * t) for t in time_steps]
    alpha_vals = [beta_vals[0]]
    for beta_i in beta_vals[1:]:
        alpha_vals.append(alpha_vals[-1] * beta_i)

    signal_rates = alpha_vals
    noise_rates = [1-a for a in alpha_vals]

    return noise_rates, signal_rates


def cosine_schedule(times:torch.Tensor) -> tuple[torch.tensor]:
    """
    times: array in range [0, 1]
    """
    theta_0 = torch.acos(0.95 * torch.ones_like(times))
    theta_1 = torch.acos(0.02 * torch.ones_like(times))

    theta_times = theta_0 + times * (theta_1 - theta_0)

    signal_rate = torch.cos(theta_times)
    noise_rate = torch.sin(theta_times)

    return signal_rate, noise_rate


def apply_noise(
    x_input: torch.Tensor,
    times: list[float]|None = None,
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
