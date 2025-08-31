from typing import Optional

import torch

from .diffusion_schedules import cosine_alpha_sigma, v_to_eps_x0


@torch.no_grad()
def inpaint_latents_vddim(
    model,
    x0_known: torch.Tensor,  # [B, C, T]
    mask: torch.Tensor,      # [B, 1, T], ones in gap
    cond: dict,
    steps: int = 30,
    cfg_scale: float = 7.0,
    clamp_mode: str = "mean",  # "mean" or "stochastic"
    device: Optional[torch.device] = None,
):
    """Simple v-DDIM sampler with clamping for known region.

    Args:
        model: U-Net model with signature forward(x_t, x_known, mask, t, **cond)
        x0_known: clean latents for the known region
        mask: binary mask (1 in gap)
        cond: dict with optional keys text_tokens, timing_tokens, clap_context
        steps: number of diffusion steps
        cfg_scale: guidance scale
        clamp_mode: 'mean' uses xi=0, 'stochastic' samples N(0, I)
    Returns:
        inpainted x0 latents [B, C, T]
    """
    device = device or x0_known.device
    B, C, T = x0_known.shape

    # Initialize x_T ~ N(0, I)
    x_t = torch.randn_like(x0_known)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t = ts[i].expand(B)
        t_next = ts[i + 1].expand(B)
        a_t, s_t = cosine_alpha_sigma(t)
        a_next, s_next = cosine_alpha_sigma(t_next)

        # Prepare known region clamping at current t
        if clamp_mode not in ("mean", "stochastic"):
            raise ValueError("clamp_mode must be 'mean' or 'stochastic'")
        if clamp_mode == "mean":
            xi = torch.zeros_like(x0_known)
        else:
            xi = torch.randn_like(x0_known)
        x_known_t = a_t.view(-1, 1, 1) * x0_known + s_t.view(-1, 1, 1) * xi
        x_t = mask * x_t + (1 - mask) * x_known_t

        # Model prediction
        def run(cond_drop=False):
            kwargs = cond.copy()
            if cond_drop:
                kwargs = kwargs.copy()
                # Drop text/timing only (classifier-free)
                kwargs["text_tokens"] = None
                kwargs["timing_tokens"] = None
            v = model(x_t, x0_known * (1 - mask), mask, t, **kwargs)
            return v

        v_cond = run(cond_drop=False)
        v_uncond = run(cond_drop=True)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        eps, x0_pred = v_to_eps_x0(x_t, v, a_t, s_t)

        # DDIM update to next t
        x_t = a_next.view(-1, 1, 1) * x0_pred + s_next.view(-1, 1, 1) * eps

    # Final x0 and merge known region
    _, x0_pred = v_to_eps_x0(x_t, v, a_next, s_next)  # type: ignore[name-defined]
    x0_inpaint = mask * x0_pred + (1 - mask) * x0_known
    return x0_inpaint

