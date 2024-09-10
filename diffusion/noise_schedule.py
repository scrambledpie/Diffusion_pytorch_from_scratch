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


if __name__=="__main__":
    time_steps = [i / 10 for i in range(10)]
    print(linear_schedule(time_steps))
