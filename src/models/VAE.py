import math

import torch.nn as nn
import torch

try:
    from snake.activations import Snake
except Exception:  # pragma: no cover - fallback if package unavailable
    class Snake(nn.Module):
        """Fallback Snake activation with trainable frequency parameter."""

        def __init__(self, in_features: int, a: float = 1.0, trainable: bool = True):
            super().__init__()
            init = torch.full((in_features,), a, dtype=torch.float32)
            if trainable:
                self.a = nn.Parameter(init)
            else:
                self.register_buffer("a", init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple formula
            return torch.where(
                self.a == 0,
                x,
                x + torch.sin(self.a * x) ** 2 / self.a
            )



class Snake1d(nn.Module):
    """Apply Snake per-channel on [B, C, T] by moving channels to the last dim."""
    def __init__(self, channels: int, a: float = 1.0, trainable: bool = True):
        super().__init__()
        self.snake = Snake(in_features=channels, a=a, trainable=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C] -> Snake -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.snake(x)
        x = x.transpose(1, 2)
        return x


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        self.snake1 = Snake1d(channels=channels, a=1.0, trainable=True)
        self.dilated_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=((kernel_size - 1) * dilation) // 2  # same length
        )
        self.snake2 = Snake1d(channels=channels, a=1.0, trainable=True)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.snake1(x)
        out = self.dilated_conv(out)
        out = self.snake2(out)
        out = self.proj(out)
        return out + residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
        downsample_stride: int = 2
    ):
        super().__init__()

        # Stack of residual dilated blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualDilatedBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=d
            ) for d in dilations
        ])
        print(f"channels: {channels}")
        # Snake activation + strided convolution for downsampling
        self.snake = Snake1d(channels=channels, a=1.0, trainable=True)
        self.downsample = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=downsample_stride,
            padding=(kernel_size - 1) // 2  # To preserve alignment
        )

    def forward(self, x):
        x = self.residual_blocks(x)

        x = self.snake(x)
        x = self.downsample(x)

        return x





class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        latent_dim: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 3,
        downsample_stride: int = 2
    ):
        super().__init__()
        print(f"hidden channels: {hidden_channels}")

        self.latent_dim = latent_dim

        # Initial projection
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

        # Encoder blocks
        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilations=(1, 2, 4),
                downsample_stride=downsample_stride
            )
            for _ in range(num_blocks)
        ])

        # Snake + two heads for mean and logvar
        self.snake = Snake1d(channels=hidden_channels, a=1.0, trainable=True)
        self.mu_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.logvar_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder_blocks(x)

        x = self.snake(x)
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)

        return mu, logvar

class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        latent_dim: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 3,
        downsample_stride: int = 2,
        decoder: nn.Module | None = None,  # plug your decoder later
    ):
        super().__init__()
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            downsample_stride=downsample_stride,
        )
        self.decoder = decoder  # optional

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps with sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode → (mu, logvar)
        mu, logvar = self.encoder(x)
        # Sample z
        z = self.reparameterize(mu, logvar)
        # Optionally decode if a decoder is provided
        if self.decoder is not None:
            x_hat = self.decoder(z)
            return x_hat, mu, logvar, z
        return mu, logvar, z

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        KL divergence between N(mu, sigma^2) and N(0, I) per element:
            0.5 * (exp(logvar) + mu^2 - 1 - logvar)
        Reduces over all dims unless 'none'.
        """
        kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        if reduction == "sum":
            return kl.sum()
        if reduction == "mean":
            return kl.mean()
        return kl  # no reduction



def compute_dilated_conv1d_output_length(
    input_length: int,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1
) -> int:
    """
    Compute the output length of a 1D dilated convolution layer.

    Parameters:
        input_length (int): Length of the input signal.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation rate. Default is 1 (standard conv).
        padding (int): Padding added to both sides. Default is 0.
        stride (int): Stride of the convolution. Default is 1.

    Returns:
        int: Output length after the convolution.
    """
    numerator = input_length + 2 * padding - dilation * (kernel_size - 1) - 1
    output_length = math.floor(numerator / stride + 1)
    return output_length




class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
        upsample_stride: int = 2
    ):
        super().__init__()
        # Snake activation + transposed convolution for upsampling
        self.snake = Snake1d(channels=channels, a=1.0, trainable=True)
        self.upsample = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=upsample_stride,
            padding=(kernel_size - 1) // 2,
            output_padding=upsample_stride - 1
        )

        # Stack of residual dilated blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualDilatedBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=d
            ) for d in dilations
        ])

    def forward(self, x):
        x = self.snake(x)
        x = self.upsample(x)
        x = self.residual_blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        hidden_channels: int = 64,
        latent_dim: int = 64,       # must match encoder latent_dim
        kernel_size: int = 3,
        num_blocks: int = 3,
        upsample_stride: int = 2
    ):
        super().__init__()

        # Project latent_dim back to hidden_channels
        self.initial_conv = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

        # Decoder blocks (reverse order of encoder)
        self.decoder_blocks = nn.Sequential(*[
            DecoderBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilations=(1, 2, 4),
                upsample_stride=upsample_stride
            )
            for _ in range(num_blocks)
        ])

        # Final projection to waveform channels
        self.final_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, z):
        """
        z: [B, latent_dim, T_latent]
        returns: waveform [B, output_channels, T_out]
        """
        x = self.initial_conv(z)
        x = self.decoder_blocks(x)
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 1, 512)  # 2 audio examples, mono, length 512
    decoder = Decoder(output_channels=1, hidden_channels=128, latent_dim=128)
    vae = VAE(input_channels=1, hidden_channels=128, latent_dim=128, decoder=decoder)

    x_recon, mu, log_var, z = vae(x)

    print("Hidden representation shape:", z.shape)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_recon.shape)
