from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..core.base_trainer import BaseTrainer
import warnings
from ..models.VAE import VAE
from ..models.conditioner import CLAPAudioConditioner
from ..utils.diffusion_schedules import cosine_alpha_sigma
import os
from tqdm import tqdm

class DiffusionTrainer(BaseTrainer):
    """
    Trainer for temporal 1-D masked inpainting diffusion (v-objective) in latent space.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = config or {}
        super().__init__(model, train_loader, val_loader, config, device, config["log_dir"])

    def _setup_training_components(self) -> None:
        # Optimizer and scheduler
        lr = self.config.get('lr', 5e-5)
        wd = self.config.get('weight_decay', 1e-3)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        warmup_steps = self.config.get('warmup_steps', 2000)
        total_steps = self.config.get('total_steps', 100_000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # cosine decay to 10% of base lr
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.55 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item() * 0.5 + 0.1

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # VAE encoder for latent space
        vae_cfg = self.config.get('vae', {
            'input_channels': 1, 'hidden_channels': 64, 'latent_dim': 64, 'kernel_size': 3, 'num_blocks': 3, 'downsample_stride': 2
        })
        self.vae = VAE(**vae_cfg).to(self.device)
        # Optionally load pretrained weights if provided
        ckpt_path = vae_cfg.get('checkpoint', None)
        if ckpt_path:
            if os.path.isfile(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state = ckpt.get('model_state_dict', ckpt)
                    missing, unexpected = self.vae.load_state_dict(state, strict=False)
                    if len(missing) > 0 or len(unexpected) > 0:
                        warnings.warn(f"Loaded VAE with missing keys: {missing} and unexpected keys: {unexpected}")
                    print(f"Loaded VAE weights from: {ckpt_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load VAE checkpoint '{ckpt_path}': {e}")
            else:
                print(f"Info: VAE checkpoint not found at '{ckpt_path}', skipping pretrained load.")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # CLAP conditioner
        clap_cfg = self.config.get('clap', None)
        self.clap = None
        if clap_cfg is not None:
            try:
                self.clap = CLAPAudioConditioner(
                    output_dim=self.config.get('cond_dim', 256),
                    clap_ckpt_path=clap_cfg['checkpoint'],
                    audio_model_type=clap_cfg.get('audio_model_type', 'HTSAT-base'),
                    enable_fusion=clap_cfg.get('enable_fusion', True),
                    project_out=True,
                ).to(self.device)
                self.clap.eval()
                for p in self.clap.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                warnings.warn(f"CLAP conditioner unavailable ({e}). Proceeding without CLAP context.")
                self.clap = None

        self.mse = nn.MSELoss(reduction='none')

    def _encode_latents(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, T] or [B,1,T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        with torch.no_grad():
            mu, logvar, z = self.vae(audio)
        # Use mu as clean latents x0
        x0 = mu
        return x0

    @staticmethod
    def _build_mask(T: int, gap_percentage: float, device: torch.device) -> torch.Tensor:
        L_gap = max(1, int(round(T * gap_percentage)))
        start = torch.randint(0, max(1, T - L_gap + 1), (1,)).item()
        end = start + L_gap
        mask = torch.zeros(1, 1, T, device=device)
        mask[:, :, start:end] = 1.0
        return mask

    def _prepare_batch(self, audio: torch.Tensor):
        # Encode
        x0 = self._encode_latents(audio.to(self.device))  # [B, C, T]
        B, C, T = x0.shape

        # Variable gap percentage support (range or fixed)
        gp = self.config.get('gap_percentage', 0.5)
        if isinstance(gp, (list, tuple)):
            gap_percentage = torch.empty(B).uniform_(gp[0], gp[1]).to(self.device)
        else:
            gap_percentage = torch.full((B,), float(gp), device=self.device)

        masks = []
        for b in range(B):
            masks.append(self._build_mask(T, gap_percentage[b].item(), self.device))
        mask = torch.cat(masks, dim=0)  # [B,1,T]

        x_known = x0 * (1 - mask)

        # Noise injection (x_t and v_target)
        # Sample t ~ U[0,1]
        t = torch.rand(B, device=self.device)
        a_t, s_t = cosine_alpha_sigma(t)
        eps = torch.randn_like(x0)
        x_t = a_t.view(-1, 1, 1) * x0 + s_t.view(-1, 1, 1) * eps
        v_target = a_t.view(-1, 1, 1) * eps - s_t.view(-1, 1, 1) * x0

        # CLAP context from waveform (never dropped)
        clap_vec = None
        if self.clap is not None:
            clap_vec = self.clap(audio if audio.dim() == 3 else audio.unsqueeze(1))  # [B, D]

        return x_t, x_known, mask, t, v_target, clap_vec

    def _compute_loss(self, v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor):
        # Masked MSE
        err = self.mse(v_pred, v_target)  # [B, C, T]
        masked = (err * mask).mean()
        lam = float(self.config.get('lambda_ctx', 0.1))
        ctx = (err * (1 - mask)).mean()
        loss = masked + lam * ctx
        return loss, masked.detach(), ctx.detach()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = total_mask = total_ctx = 0.0
        # AMP scaler (new API); fall back works if disabled
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


        pbar = tqdm(
            total=len(self.train_loader),
            desc='Training UNET for Diffusion',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )
        # Zero grad once at start of accumulation window
        self.optimizer.zero_grad(set_to_none=True)
        step_interval = max(1, int(self.config.get('step_interval', 1)))
        for batch_idx, batch in enumerate(self.train_loader):
            audio = batch.to(self.device)  # [B, T]
            x_t, x_known, mask, t, v_target, clap_vec = self._prepare_batch(audio)

            # Autocast using new API when available
            try:
                autocast_ctx = torch.amp.autocast(device_type, enabled=(device_type == 'cuda'))
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())

            with autocast_ctx:
                v_pred = self.model(
                    x_t,
                    x_known,
                    mask,
                    t,
                    clap_context=clap_vec,
                    text_tokens=None,
                    timing_tokens=None,
                    cfg_dropout_p=self.config.get('cfg_dropout_p', 0.0),
                )
                loss, masked, ctx = self._compute_loss(v_pred, v_target, mask)

            # Backprop with accumulation; optimizer step at intervals
            scaler.scale(loss / step_interval).backward()
            do_step = ((batch_idx + 1) % step_interval == 0) or ((batch_idx + 1) == len(self.train_loader))
            if do_step:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()
            total_mask += masked.item()
            total_ctx += ctx.item()
            pbar.set_postfix(loss=loss.item(), masked_loss=masked.item(), context_loss=ctx.item())
            pbar.update(1)

            self.global_step += 1
            self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
            self.writer.add_scalar('train/masked_loss_step', masked.item(), self.global_step)
            self.writer.add_scalar('train/context_loss_step', ctx.item(), self.global_step)

        n = max(1, len(self.train_loader))
        return {
            'loss': total_loss / n,
            'masked_loss': total_mask / n,
            'context_loss': total_ctx / n,
        }

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": 0.0}
        self.model.eval()
        total_loss = total_mask = total_ctx = 0.0
        pbar = tqdm(
            total=len(self.val_loader),
            desc='Validating UNET for Diffusion',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )
        for batch in self.val_loader:
            audio = batch.to(self.device)
            x_t, x_known, mask, t, v_target, clap_vec = self._prepare_batch(audio)
            v_pred = self.model(
                x_t,
                x_known,
                mask,
                t,
                clap_context=clap_vec,
                text_tokens=None,
                timing_tokens=None,
                cfg_dropout_p=0.0,
            )
            loss, masked, ctx = self._compute_loss(v_pred, v_target, mask)
            total_loss += loss.item()
            total_mask += masked.item()
            total_ctx += ctx.item()
            pbar.set_postfix(loss=loss.item(), masked_loss=masked.item(), context_loss=ctx.item())
            pbar.update(1)

        n = max(1, len(self.val_loader))
        return {
            'loss': total_loss / n,
            'masked_loss': total_mask / n,
            'context_loss': total_ctx / n,
        }
