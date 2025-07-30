"""
Patch shuffle component for MAE (Masked Autoencoder) models.
"""
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, Optional
from einops import repeat


def random_indexes(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random forward and backward indexes for shuffling."""
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """Take sequences at specified indexes."""
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    """
    Patch shuffle component for MAE models.
    
    This module handles the masking and shuffling of patches in a spectrogram,
    implementing the masking strategy used in Masked Autoencoders.
    """
    
    def __init__(self, ratio: float, num_rows: int, num_cols: int) -> None:
        """
        Initialize the patch shuffle module.
        
        Args:
            ratio: Fraction of columns to mask (0.75 => mask 75% of columns)
            num_rows: Patch-grid height (H // patch_size)
            num_cols: Patch-grid width (W // patch_size)
        """
        super().__init__()
        self.ratio = ratio
        self.num_rows = num_rows
        self.num_cols = num_cols
    
    def forward(
        self,
        patches: torch.Tensor,
        bounds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for patch shuffling.

        Args:
            patches: Input patches with shape ``[T, B, C]`` where ``T = num_rows * num_cols``.
            bounds: Optional tensor specifying the start and end column (in patch
                indices) to mask for each sample. Shape should be ``[2, B]``.
                If ``None``, a random start column is used.

        Returns:
            Tuple containing:
                - Shuffled patches
                - Forward indexes
                - Backward indexes
                - Stripe bounds
        """
        T, B, C = patches.shape
        r, c = self.num_rows, self.num_cols
        device = patches.device
        
        remain_list = []
        fwd_list, bwd_list, bounds_list = [], [], []
        
        stripe_width = max(1, int(c * self.ratio))  # columns to mask
        
        # Pre-compute a (r, c) grid of flattened indices
        grid = torch.arange(T, device=device).view(r, c)  # shape [r, c]
        
        for b in range(B):
            if bounds is not None:
                start_col = int(bounds[0, b].item())
                end_col = int(bounds[1, b].item())
                start_col = max(0, min(start_col, c - 1))
                end_col = max(start_col + 1, min(end_col, c))
            else:
                start_col = random.randint(0, c - stripe_width)
                end_col = start_col + stripe_width
            
            # Visible = columns outside stripe
            visible_cols_left = grid[:, :start_col].flatten()
            visible_cols_right = grid[:, end_col:].flatten()
            visible = torch.cat([visible_cols_left, visible_cols_right], dim=0)
            
            # Masked = columns inside stripe
            masked = grid[:, start_col:end_col].flatten()
            
            # Forward index order = visible first, then masked
            fwd = torch.cat([visible, masked], dim=0)
            
            # Backward map (inverse permutation)
            bwd = torch.argsort(fwd)
            
            remain_list.append(len(visible))
            fwd_list.append(fwd)
            bwd_list.append(bwd)
            bounds_list.append(torch.tensor([start_col, end_col], device=device))
        
        # Stack per-batch index tensors → [T, B]
        forward_indexes = torch.stack(fwd_list, dim=-1)
        backward_indexes = torch.stack(bwd_list, dim=-1)
        stripe_bounds = torch.stack(bounds_list, dim=-1)  # shape [2, B]
        
        # Gather patches so visible tokens come first
        patches = take_indexes(patches, forward_indexes)
        
        # Keep only visible part (they might differ per sample; take min)
        min_visible = min(remain_list)
        patches = patches[:min_visible]
        
        return patches, forward_indexes, backward_indexes, stripe_bounds 