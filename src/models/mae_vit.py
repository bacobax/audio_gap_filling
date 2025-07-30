"""
MAE (Masked Autoencoder) Vision Transformer implementation.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import Block
from typing import Tuple, Optional
import random

from ..core.base_model import BaseModel
from .patch_shuffle import PatchShuffle
from transformers import AutoModel

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class MAEEncoder(nn.Module):
    """
    MAE Encoder component.
    
    This module encodes the visible patches of a masked spectrogram using
    a Vision Transformer architecture.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (80, 380),
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 12,
        num_head: int = 3,
        mask_ratio: float = 0.75,
        pretrained_ViT= False
    ) -> None:
        """
        Initialize the MAE encoder.
        
        Args:
            image_size: Size of input spectrogram (height, width)
            patch_size: Size of patches
            emb_dim: Embedding dimension
            num_layer: Number of transformer layers
            num_head: Number of attention heads
            mask_ratio: Ratio of patches to mask
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size), 1, emb_dim)
        )
        self.num_rows = image_size[0] // patch_size
        self.num_cols = image_size[1] // patch_size
        self.shuffle = PatchShuffle(mask_ratio, self.num_rows, self.num_cols)
        
        self.patchify = nn.Conv2d(1, emb_dim, patch_size, patch_size, bias=False)
        print(f"using pretrained vit: {pretrained_ViT}")
        if pretrained_ViT:
            print("Using pretrined AUDIOMAE")
            full_model = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer = full_model.encoder.blocks
            self.layer_norm = full_model.encoder.norm
        else:
            self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
            self.layer_norm = nn.LayerNorm(emb_dim)
        self.patch2img = Rearrange(
            '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size,
            p2=patch_size,
            h=image_size[0] // patch_size,
            w=image_size[1] // patch_size
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
    
    def forward(
        self,
        img: torch.Tensor,
        mask_bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            img: Input spectrogram.
            mask_bounds: Optional tuple ``(start, end)`` specifying the start and
                end column (in pixel indices) of an already-masked region. If
                ``None``, a random region will be masked.

        Returns:
            Tuple containing:
                - Encoded features
                - Backward indexes
                - Masked image corresponding to ``mask_bounds``
        """
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        if mask_bounds is not None:
            start_pix, end_pix = mask_bounds
            start_patch = start_pix // self.patch_size
            end_patch = (end_pix + self.patch_size - 1) // self.patch_size
            bounds = torch.tensor([[start_patch], [end_patch]], device=patches.device).repeat(1, patches.shape[1])
            patches, forward_indexes, backward_indexes, stripe_bounds = self.shuffle(patches, bounds)
        else:
            patches, forward_indexes, backward_indexes, stripe_bounds = self.shuffle(patches)
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        
        # Build full-width masked image with zeros in the gap
        B = img.shape[0]
        masked_full = torch.zeros_like(img)  # [B, 1, 80, 380]
        for b in range(B):
            s, e = stripe_bounds[:, b]  # start & end column (patch idx)
            # Convert patch columns to pixel columns
            s_pix = int(s * self.patch_size)
            e_pix = int(e * self.patch_size)
            masked_full[b, :, :, :s_pix] = img[b, :, :, :s_pix]
            masked_full[b, :, :, e_pix:] = img[b, :, :, e_pix:]
        
        return features, backward_indexes, masked_full


class MAEDecoder(nn.Module):
    """
    MAE Decoder component.
    
    This module reconstructs the full spectrogram from encoded features.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (80, 380),
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 4,
        num_head: int = 3,
    ) -> None:
        """
        Initialize the MAE decoder.
        
        Args:
            image_size: Size of output spectrogram (height, width)
            patch_size: Size of patches
            emb_dim: Embedding dimension
            num_layer: Number of transformer layers
            num_head: Number of attention heads
        """
        super().__init__()
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size) + 1, 1, emb_dim)
        )
        
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = nn.Linear(emb_dim, patch_size ** 2)
        self.patch2img = Rearrange(
            '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size,
            p2=patch_size,
            h=image_size[0] // patch_size,
            w=image_size[1] // patch_size
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
    
    def forward(self, features: torch.Tensor, backward_indexes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder.
        Args:
            features: Encoded features from the encoder
            backward_indexes: Backward indexes for reconstruction
        Returns:
            Tuple containing:
                - Reconstructed spectrogram
                - Mask indicating which patches were masked
        """
        T = features.shape[0]

        backward_indexes = torch.cat([
            torch.zeros(1, backward_indexes.shape[1], dtype=torch.long, device=backward_indexes.device),
            backward_indexes + 1
        ], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = torch.gather(features, 0, backward_indexes.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MAEViT(BaseModel):
    """
    MAE (Masked Autoencoder) Vision Transformer model.
    
    This model implements the MAE architecture for spectrogram reconstruction,
    using a Vision Transformer encoder-decoder with masking.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the MAE ViT model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)
        
        # Extract configuration
        image_size = config.get('image_size', (80, 380))
        patch_size = config.get('patch_size', 2)
        emb_dim = config.get('emb_dim', 192)
        encoder_layer = config.get('encoder_layer', 12)
        encoder_head = config.get('encoder_head', 3)
        decoder_layer = config.get('decoder_layer', 4)
        decoder_head = config.get('decoder_head', 3)
        mask_ratio = config.get('mask_ratio', 0.75)
        pretrained_ViT = config.get('pretrained_ViT', False)
        
        # Create encoder and decoder
        self.encoder = MAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
            pretrained_ViT=pretrained_ViT
        )
        
        self.decoder = MAEDecoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            num_head=decoder_head
        )
        
        self.mask_ratio = mask_ratio
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = ['image_size', 'patch_size', 'emb_dim']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate patch size compatibility
        image_size = self.config['image_size']
        patch_size = self.config['patch_size']
        
        # if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
        #     raise ValueError(
        #         f"Image size {image_size} must be divisible by patch size {patch_size}"
        #     )
    
    def forward(
        self,
        img: torch.Tensor,
        mask_bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MAE model.

        Args:
            img: Input spectrogram.
            mask_bounds: Optional tuple ``(start, end)`` describing the region to
                mask (in pixel indices). If ``None``, a random region is masked.

        Returns:
            Tuple containing:
                - Reconstructed spectrogram
                - Mask indicating which patches were masked
        """
        features, backward_indexes, _ = self.encoder(img, mask_bounds=mask_bounds)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask