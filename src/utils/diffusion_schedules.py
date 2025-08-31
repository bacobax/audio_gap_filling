import math
import torch


def cosine_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine schedule producing (alpha_t, sigma_t) with t in [0,1].

    Uses the schedule from Nichol & Dhariwal (improved DDPM) adapted to v-objective.
    """
    s = 0.008
    t_ = (t + s) / (1 + s)
    alphas_cumprod = torch.cos(t_ * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alpha = torch.sqrt(alphas_cumprod).clamp(0.0, 1.0)
    sigma = torch.sqrt(1 - alphas_cumprod).clamp(0.0, 1.0)
    return alpha, sigma


def v_to_eps_x0(x_t: torch.Tensor, v: torch.Tensor, alpha_t: torch.Tensor, sigma_t: torch.Tensor):
    """Recover epsilon and x0 from v-parameterization.

    x0 = alpha * x_t - sigma * v
    eps = sigma * x_t + alpha * v
    """
    # reshape for broadcasting: [B] -> [B,1,1]
    a = alpha_t.view(-1, 1, 1)
    s = sigma_t.view(-1, 1, 1)
    x0 = a * x_t - s * v
    eps = s * x_t + a * v
    return eps, x0

