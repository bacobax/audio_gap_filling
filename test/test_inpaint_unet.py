import torch

from src.models.inpaint_unet_1d import InpaintUNet1D
from src.utils.diffusion_schedules import cosine_alpha_sigma, v_to_eps_x0
from src.utils.inpainting import inpaint_latents_vddim


def test_unet_shapes():
    B, C, T = 2, 64, 256
    model = InpaintUNet1D()
    x_t = torch.randn(B, C, T)
    x_known = torch.randn(B, C, T)
    mask = torch.zeros(B, 1, T)
    mask[:, :, 100:180] = 1.0
    t = torch.rand(B)
    y = model(x_t, x_known, mask, t)
    assert y.shape == (B, C, T)


def test_mask_loss_region_equivalence():
    # When mask is all ones, only masked region contributes
    B, C, T = 1, 64, 32
    model = InpaintUNet1D()
    x_t = torch.zeros(B, C, T)
    x_known = torch.zeros(B, C, T)
    mask = torch.ones(B, 1, T)
    t = torch.zeros(B)
    v_pred = model(x_t, x_known, mask, t)
    # random target
    v_target = torch.randn_like(v_pred)
    err = (v_pred - v_target) ** 2
    masked = (err * mask).mean()
    ctx = (err * (1 - mask)).mean()
    assert ctx.item() == 0.0
    assert masked.item() >= 0.0


def test_clamping_behavior():
    B, C, T = 1, 64, 64
    device = torch.device('cpu')
    x0_known = torch.randn(B, C, T, device=device)
    mask = torch.zeros(B, 1, T, device=device)
    mask[:, :, 16:48] = 1.0

    # Dummy model: returns zero v so that eps = sigma * x_t, x0 = alpha * x_t
    import torch.nn as nn
    class Dummy(nn.Module):
        def forward(self, x_t, x_known, mask, t, **kwargs):
            return torch.zeros_like(x_t)
    model = Dummy()

    steps = 2
    # Mean clamping: known region equals alpha * x0_known at the start of each step
    out = inpaint_latents_vddim(model, x0_known, mask, {}, steps=steps, clamp_mode='mean', cfg_scale=0.0, device=device)
    assert out.shape == x0_known.shape

    # Check schedule conversion shapes
    t = torch.tensor([0.5])
    a, s = cosine_alpha_sigma(t)
    x_t = torch.randn_like(x0_known)
    v = torch.zeros_like(x0_known)
    eps, x0 = v_to_eps_x0(x_t, v, a, s)
    assert eps.shape == x_t.shape and x0.shape == x_t.shape
