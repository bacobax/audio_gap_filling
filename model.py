from dataset import MelSpecRandomCropDataset
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from utils import MCD
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio, num_rows=None, num_cols=None) -> None:
        """
        ratio     : fraction of columns to mask (0.75 => mask 75 % of columns)
        num_rows  : patch-grid height  (H // patch_size)
        num_cols  : patch-grid width   (W // patch_size)
        """
        super().__init__()
        self.ratio = ratio
        self.num_rows = num_rows
        self.num_cols = num_cols

    def forward(self, patches: torch.Tensor):
        """
        patches shape: [T, B, C] with T = num_rows * num_cols
        We mask a contiguous range of columns for every sample in the batch.
        """
        T, B, C = patches.shape
        r, c = self.num_rows, self.num_cols
        device = patches.device

        remain_list = []
        fwd_list, bwd_list, bounds_list = [], [], []

        stripe_width = max(1, int(c * self.ratio))  # columns to mask

        # Pre‑compute a (r, c) grid of flattened indices
        grid = torch.arange(T, device=device).view(r, c)  # shape [r, c]

        for _ in range(B):
            # ── choose random start col ───────────────────────────────────────
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

        # Stack per‑batch index tensors → [T, B]
        forward_indexes = torch.stack(fwd_list, dim=-1)
        backward_indexes = torch.stack(bwd_list, dim=-1)
        stripe_bounds = torch.stack(bounds_list, dim=-1)   # shape [2, B]

        # Gather patches so visible tokens come first
        patches = take_indexes(patches, forward_indexes)

        # Keep only visible part (they might differ per sample; take min)
        min_visible = min(remain_list)
        patches = patches[:min_visible]

        return patches, forward_indexes, backward_indexes, stripe_bounds

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=(80, 380),
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.patch_size=patch_size
        self.image_size=image_size
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size), 1, emb_dim))
        self.num_rows = image_size[0] // patch_size
        self.num_cols = image_size[1] // patch_size
        self.shuffle = PatchShuffle(mask_ratio, self.num_rows, self.num_cols)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size, bias=False)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size[0]//patch_size, w=image_size[1]//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, apply_mask=True):
        if not apply_mask:
            patches = self.patchify(img)
            patches = rearrange(patches, 'b c h w -> (h w) b c')
            patches = patches + self.pos_embedding
            patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
            patches = rearrange(patches, 't b c -> b t c')
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, 'b t c -> t b c')
            return features, None, img

        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes, stripe_bounds = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        # Build full‑width masked image with zeros in the gap
        B = img.shape[0]
        masked_full = torch.zeros_like(img)            # [B, 1, 80, 380]
        for b in range(B):
            s, e = stripe_bounds[:, b]                 # start & end column (patch idx)
            # Convert patch columns to pixel columns
            s_pix = int(s * self.patch_size)
            e_pix = int(e * self.patch_size)
            masked_full[b, :, :, :s_pix] = img[b, :, :, :s_pix]
            masked_full[b, :, :, e_pix:] = img[b, :, :, e_pix:]

        return features, backward_indexes, masked_full

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=(80, 380),
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, patch_size ** 2)
        self.patch2img = Rearrange(
            '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size,
            p2=patch_size,
            h=image_size[0]//patch_size,
            w=image_size[1]//patch_size
        )

        self.init_weight()

    # ──────────────────────────────────────────────────────────────────────
    def patches_to_image(self, patch_tokens: torch.Tensor):
        """
        Reconstruct the spatial image from patch tokens.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            Shape: [T, B, 3 * patch_size ** 2]
            where T = (H / patch_size) * (W / patch_size).

        Returns
        -------
        img : torch.Tensor
            Shape: [B, 3, H, W]
        """
        # Use the same Rearrange layer defined in __init__
        return self.patch2img(patch_tokens)

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        if backward_indexes is None:
            # No masking, so identity permutation
            # T = number of tokens, B = batch size
            T, B, _ = features.shape
            backward_indexes = torch.arange(T, device=features.device).unsqueeze(-1).repeat(1, B)
        else:
            backward_indexes = torch.cat([
                torch.zeros(1, backward_indexes.shape[1], dtype=torch.long, device=backward_indexes.device),
                backward_indexes + 1
            ], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=(80, 380),
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img, apply_mask=True):
        features, backward_indexes, full_mask = self.encoder(img, apply_mask=apply_mask)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


# if __name__ == '__main__':
#     shuffle = PatchShuffle(0.02, num_rows=2, num_cols=8)
#     a = torch.rand(16, 2, 2)
#     b, forward_indexes, backward_indexes,_ = shuffle(a)
#     print(b.shape)
#     mask_ratio = 0.25
#     img = torch.rand(2, 1, 80, 380)
#     encoder = MAE_Encoder(image_size=(80, 380), mask_ratio=0.25)
#     decoder = MAE_Decoder(image_size=(80, 380))
#     features, backward_indexes, masked_full = encoder(img)
#     print(forward_indexes.shape)
#     predicted_img, mask = decoder(features, backward_indexes)
#     print(predicted_img.shape)
#     loss = torch.mean((predicted_img - img) ** 2 * mask / mask_ratio)
#     print(loss)

#     # masked_full: [B, C, H, W] – show first sample & first channel
#     plt.imshow(masked_full[0, 0].detach().cpu().numpy(), aspect="auto",
#                origin="lower", cmap="viridis")
#     plt.colorbar(label="Magnitude (arbitrary units)")
#     plt.title("Masked Spectrogram Heatmap")
#     plt.tight_layout()
#     plt.show()

#     # Also visualise just the gap (boolean mask)
#     gap_mask = (masked_full[0, 0] == 0).float()  # 1 where gap, 0 elsewhere
#     plt.figure(figsize=(10, 2))
#     plt.imshow(gap_mask.numpy(), aspect="auto", origin="lower", cmap="gray")
#     plt.title("Gap Mask")
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':


    n_mels = 80

    mask_ratio = 0.25
    patch_size = 4

    min_patch_size=1
    max_patch_size=80

    mydataset = MelSpecRandomCropDataset(
        gap_percentage=mask_ratio,
        flac_path="gapped_audio.wav",
        hop_length=256,
        n_fft=1024,
        n_mels=n_mels,
        test=False
    )

    crop_frames = mydataset.crop_frames
    print(f"crop frames: {crop_frames}")
    dataloader = DataLoader(mydataset, batch_size=1, shuffle=True)


    if not crop_frames % patch_size == 0 and n_mels % patch_size == 0:
        common_divisors = MCD(crop_frames, n_mels, min_patch_size,max_patch_size)
        if(len(common_divisors) == 0):
            raise Exception(f"No common divisors between {crop_frames} and {n_mels} in the range ({min_patch_size}, {max_patch_size})")
        else:
            raise Exception(f"crop frames {crop_frames} and n_mels {n_mels} not divisible by the patch size {patch_size}, try these other patch sizes: ", *common_divisors)

    encoder = MAE_Encoder(image_size=(n_mels, crop_frames), mask_ratio=mask_ratio, patch_size=patch_size)
    decoder = MAE_Decoder(image_size=(n_mels, crop_frames), patch_size=patch_size)

    # Conditional inference block without masking
    test_sample = next(iter(dataloader))
    spectrogram_slice, start_idx, end_idx, start_sec, end_sec, _, _, start_gap_sec, end_gap_sec = test_sample
    features, _, _ = encoder(spectrogram_slice, apply_mask=False)
    predicted_spectrogram_slice, _ = decoder(features, None)

    plt.imshow(predicted_spectrogram_slice.squeeze().detach().cpu().numpy(), aspect="auto",
               origin="lower", cmap="viridis")
    plt.colorbar(label="Magnitude (arbitrary units)")
    plt.title("Final Prediction on Gapped Input")
    plt.tight_layout()
    plt.show()


    print(f"dataloader len: {len(dataloader)}")
    for i, (spectrogram_slice, start_idx, end_idx, start_sec, end_sec, _, _, start_gap_sec, end_gap_sec) in enumerate(dataloader):
        print(f"REAL GAP: [{start_gap_sec}, {end_gap_sec}]")
        start_sec, end_sec = start_sec.item(), end_sec.item()
        duration = end_sec - start_sec
        features, backward_indexes, masked_full = encoder(spectrogram_slice)
        predicted_spectrogram_slice, mask = decoder(features, backward_indexes)
        print(predicted_spectrogram_slice.shape)
        loss = torch.mean((predicted_spectrogram_slice - spectrogram_slice) ** 2 * mask / mask_ratio)
        print(loss)

        # masked_full: [B, C, H, W] – show first sample & first channel
        plt.imshow(masked_full[0, 0].detach().cpu().numpy(), aspect="auto",
                origin="lower", cmap="viridis")
        plt.colorbar(label="Magnitude (arbitrary units)")
        plt.title(f"Masked Spectrogram Heatmap ({start_sec}, {end_sec}) duration: {duration}")
        plt.tight_layout()
        plt.show()

         # masked_full: [B, C, H, W] – show first sample & first channel
        plt.imshow(predicted_spectrogram_slice.squeeze().detach().cpu().numpy(), aspect="auto",
                origin="lower", cmap="viridis")
        plt.colorbar(label="Magnitude (arbitrary units)")
        plt.title(f"Masked Spectrogram Heatmap ({start_sec}, {end_sec}) duration: {duration}")
        plt.tight_layout()
        plt.show()

        reconstructed = mydataset.reconstruct_spectrogram(predicted_spectrogram_slice, start_idx, end_idx)
        plt.imshow(reconstructed, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Magnitude (arbitrary units)")
        plt.title(f"full reconstructed spectrogram ({start_sec}, {end_sec}) duration: {duration}")
        plt.tight_layout()
        plt.show()

        # Also visualise just the gap (boolean mask)
        gap_mask = (masked_full[0, 0] == 0).float()  # 1 where gap, 0 elsewhere
        plt.figure(figsize=(10, 2))
        plt.imshow(gap_mask.numpy(), aspect="auto", origin="lower", cmap="gray")
        plt.title("Gap Mask")
        plt.tight_layout()
        plt.show()
