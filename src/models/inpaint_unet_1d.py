import math
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F


def get_sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: float tensor of shape [B] in [0, 1]
        dim: embedding dimension
    Returns:
        [B, dim]
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        * -(math.log(10_000.0) / (half_dim - 1))
    )
    # [B, 1] * [H] -> [B, H]
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Produce FiLM parameters (gamma, beta) from conditioning embedding."""

    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_channels * 2),
        )

    def forward(self, h: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        ab = self.net(h)
        a, b = ab.chunk(2, dim=-1)
        return a, b


class ResidualBlock1D(nn.Module):
    """Residual block with GroupNorm, SiLU, Conv1d and FiLM modulation."""

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.film = FiLM(cond_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, cond_h: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        gamma, beta = self.film(cond_h)
        # FiLM: broadcast over time
        x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        qkv = self.qkv(x)  # [B, 3C, T]
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to heads: [B, heads, dim, T]
        q = q.view(B, self.n_heads, self.head_dim, T)
        k = k.view(B, self.n_heads, self.head_dim, T)
        v = v.view(B, self.n_heads, self.head_dim, T)
        attn = torch.einsum('bhdt,bhds->bhst', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhst,bhds->bhdt', attn, v).contiguous()
        out = out.view(B, C, T)
        return self.proj(out)


class CrossAttention1D(nn.Module):
    """Cross attention over time with conditioning tokens.

    Q from latent features (pooled over channels), K/V from conditioning tokens.
    """

    def __init__(self, channels: int, cond_dim: int, n_heads: int = 8):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Linear(cond_dim, channels)
        self.v = nn.Linear(cond_dim, channels)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], tokens: [B, S, D]
        B, C, T = x.shape
        S = tokens.shape[1]
        q = self.q(x)  # [B, C, T]
        q = q.view(B, self.n_heads, self.head_dim, T)

        k = self.k(tokens).view(B, S, self.n_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D, S]
        v = self.v(tokens).view(B, S, self.n_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, H, D, S]

        attn = torch.einsum('bhdt,bhds->bhst', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhst,bhds->bhdt', attn, v).contiguous()
        out = out.view(B, C, T)
        return self.proj(out)


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class InpaintUNet1D(nn.Module):
    """
    Temporal 1-D U-Net for masked inpainting diffusion in latent space.

    Predicts v on the gap region only, conditioned on:
    - clean left/right context latents (x_known)
    - binary gap mask (1 in gap)
    - timestep embedding
    - optional text/timing tokens
    - CLAP context vector (used for FiLM and as a token)
    """

    def __init__(
        self,
        in_channels: int = 129,  # 64 (x_t) + 64 (x_known) + 1 (mask)
        latents_channels: int = 64,
        channels_per_scale: tp.Sequence[int] = (128, 256, 384, 512, 768),
        self_attn_scales: tp.Sequence[int] = (3, 4),  # indices of scales where to use self-attn
        num_res_blocks: int = 2,
        cond_dim: int = 256,  # internal conditioning embedding size
        n_heads: int = 8,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.latents_channels = latents_channels
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # First projection after concatenation
        c0 = channels_per_scale[0]
        self.input_proj = nn.Conv1d(in_channels, c0, kernel_size=3, padding=1)

        # Time embedding MLP (sinusoidal -> MLP)
        self.t_embed_dim = cond_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4), nn.SiLU(), nn.Linear(cond_dim * 4, cond_dim)
        )

        # Gate for optional text/timing/CLAP condition vectors
        # We'll combine: pooled_text + pooled_timing + clap_context -> cond vector
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim * 3, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        # Down / Up paths
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        prev_c = c0
        skip_channels = []
        n_scales = len(channels_per_scale)
        for i, c in enumerate(channels_per_scale):
            # Current scale channels are prev_c (equals channels_per_scale[i])
            res = nn.ModuleList([ResidualBlock1D(prev_c, cond_dim) for _ in range(num_res_blocks)])
            self.down_blocks.append(res)
            # cross-attn per scale
            self.cross_attn.append(CrossAttention1D(prev_c, cond_dim, n_heads=n_heads))
            # optional self-attn at coarse scales
            if i in self_attn_scales:
                self.self_attn.append(SelfAttention1D(prev_c, n_heads=n_heads))
            else:
                self.self_attn.append(nn.Identity())
            skip_channels.append(prev_c)
            if i < n_scales - 1:
                next_c = channels_per_scale[i + 1]
                self.downsamples.append(Downsample1D(prev_c, next_c))
                prev_c = next_c

        # Bottleneck
        self.bottleneck = nn.ModuleList([ResidualBlock1D(prev_c, cond_dim) for _ in range(num_res_blocks)])
        self.bottleneck_attn = SelfAttention1D(prev_c, n_heads=n_heads)

        # Upsampling
        for i in reversed(range(len(channels_per_scale))):
            c = skip_channels[i]  # target channels after upsample and for skip
            # Residual blocks operate after concatenation: channels = c (upsampled) + c (skip) = 2c
            block = nn.ModuleList([ResidualBlock1D(c + c, cond_dim) for _ in range(num_res_blocks)])
            self.up_blocks.append(block)
            self.upsamples.append(Upsample1D(prev_c, c))
            self.self_attn.append(nn.Identity())  # placeholder to keep indexing aligned
            prev_c = c + c

        # Final projection to latents channels
        self.out = nn.Sequential(
            nn.GroupNorm(8, prev_c), nn.SiLU(), nn.Conv1d(prev_c, latents_channels, kernel_size=3, padding=1)
        )

        # Project context vector into a token for cross-attention
        self.clap_token_proj = nn.Linear(cond_dim, cond_dim)

    def _maybe_ckpt(self, fn, *args):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(
        self,
        x_t: torch.Tensor,  # [B, C=latents_channels, T]
        x_known: torch.Tensor,  # [B, C=latents_channels, T]
        mask: torch.Tensor,  # [B, 1, T], ones in gap
        t: torch.Tensor,  # [B] in [0,1]
        text_tokens: tp.Optional[torch.Tensor] = None,  # [B, S1, D]
        timing_tokens: tp.Optional[torch.Tensor] = None,  # [B, S2, D]
        clap_context: tp.Optional[torch.Tensor] = None,  # [B, D]
        cfg_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        B, C, T = x_t.shape

        # Concatenate inputs
        x_in = torch.cat([x_t, x_known, mask.expand(-1, 1, -1)], dim=1)
        h = self.input_proj(x_in)

        # Timestep embedding
        t_emb = get_sinusoidal_timestep_embedding(t, self.t_embed_dim)
        t_emb = self.t_mlp(t_emb)

        # Prepare conditioning tokens and pooled embeddings
        if text_tokens is not None and cfg_dropout_p > 0:
            keep = (torch.rand(B, device=x_t.device) > cfg_dropout_p).float().view(B, 1, 1)
            text_tokens = text_tokens * keep
        if timing_tokens is not None and cfg_dropout_p > 0:
            keep = (torch.rand(B, device=x_t.device) > cfg_dropout_p).float().view(B, 1, 1)
            timing_tokens = timing_tokens * keep

        pooled_text = (
            text_tokens.mean(dim=1) if (text_tokens is not None and text_tokens.numel() > 0) else torch.zeros(B, self.t_embed_dim, device=x_t.device)
        )
        pooled_timing = (
            timing_tokens.mean(dim=1) if (timing_tokens is not None and timing_tokens.numel() > 0) else torch.zeros(B, self.t_embed_dim, device=x_t.device)
        )
        pooled_clap = (
            clap_context if clap_context is not None else torch.zeros(B, self.t_embed_dim, device=x_t.device)
        )

        cond_vec = self.cond_proj(torch.cat([t_emb, pooled_timing, pooled_text], dim=-1))
        # Include CLAP in cond via residual add (not dropped)
        cond_vec = cond_vec + pooled_clap

        # Build cross-attention token sequence: [text, timing, clap_token]
        tokens: tp.List[torch.Tensor] = []
        if text_tokens is not None:
            tokens.append(text_tokens)
        if timing_tokens is not None:
            tokens.append(timing_tokens)
        # CLAP token
        clap_tok = self.clap_token_proj(pooled_clap).unsqueeze(1)  # [B,1,D]
        tokens.append(clap_tok)
        cross_tokens = torch.cat(tokens, dim=1) if len(tokens) > 0 else clap_tok

        # Down path with skips
        skips = []
        for res_blocks, ca, sa, down in zip(self.down_blocks, self.cross_attn, self.self_attn, self.downsamples + [None]):
            for rb in res_blocks:
                h = self._maybe_ckpt(rb, h, cond_vec)
            h = h + ca(h, cross_tokens)
            h = h + (sa(h) if not isinstance(sa, nn.Identity) else h * 0)  # no-op if identity
            skips.append(h)
            if down is not None:
                h = down(h)

        # Bottleneck
        for rb in self.bottleneck:
            h = self._maybe_ckpt(rb, h, cond_vec)
        h = h + self.bottleneck_attn(h)

        # Up path
        for up, upsample, skip in zip(self.up_blocks, self.upsamples, reversed(skips)):
            h = upsample(h)
            # match time length if off by 1 due to stride/rounding
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            for rb in up:
                h = self._maybe_ckpt(rb, h, cond_vec)

        v_pred = self.out(h)
        return v_pred
