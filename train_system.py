
from dataset import MelSpecRandomCropDataset
from model import MAE_ViT
import math, os, torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np

from utils import MCD
from datetime import datetime


class TrainSystem:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.run_name = f"mae-pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(os.path.join('runs', 'test', self.run_name))
        
        self.batch_size = self.hparams.get('batch_size', 4096)

        self.mask_ratio = self.hparams.get('mask_ratio', 0.25)
        self.patch_size = self.hparams.get('patch_size', 10)
        self.n_mels = self.hparams.get('n_mels', 80)
        self.audio_filename = self.hparams.get('audio_filename', 'gapped_audio.wav')

        train_dataset = MelSpecRandomCropDataset(
            gap_percentage=self.mask_ratio,
            flac_path=self.audio_filename,
            hop_length=256,
            n_fft=1024,
            n_mels=self.n_mels
        )
        self.val_dataset = MelSpecRandomCropDataset(
            gap_percentage=self.mask_ratio,
            flac_path=self.audio_filename,
            hop_length=256,
            n_fft=1024,
            n_mels=self.n_mels,
            test=(True, self.hparams.get('test_audio_filename', 'wav_test.wav'))
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        self.crop_frames = train_dataset.crop_frames
        self.check_patch_compatibility(train_dataset)

        load_batch_size = min(self.hparams.get('max_device_batch_size', 512), self.batch_size)
        assert self.batch_size % load_batch_size == 0
        self.steps_per_update = self.batch_size // load_batch_size

        self.dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
        self.model = MAE_ViT(
            image_size=(self.n_mels, self.crop_frames),
            patch_size=self.patch_size,
            emb_dim=hparams.get("emb_dim", 192),
            encoder_layer=hparams.get("encoder_layer", 12),
            encoder_head=hparams.get("encoder_head", 3),
            decoder_layer=hparams.get("decoder_layer", 4),
            decoder_head=hparams.get("decoder_head", 3),
            mask_ratio=self.mask_ratio,
        )
        self.model = self.model.to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.get('base_learning_rate', 1.5e-4) * self.batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.get('weight_decay', 0.05)
        )

    def check_patch_compatibility(self, mydataset):
        crop_frames = mydataset.crop_frames
        min_patch_size = 1
        max_patch_size = 80
        if not crop_frames % self.patch_size == 0 and self.n_mels % self.patch_size == 0:
            common_divisors = MCD(crop_frames, self.n_mels, min_patch_size, max_patch_size)
            if len(common_divisors) == 0:
                raise Exception(
                    f"No common divisors between {crop_frames} and {self.n_mels} in the range ({min_patch_size}, {max_patch_size})")
            else:
                raise Exception(
                    f"crop frames {crop_frames} and n_mels {self.n_mels} not divisible by the patch size {self.patch_size}, try these other patch sizes: ",
                    *common_divisors)

    def train(self):

        def setup_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        args = self.hparams

        setup_seed(args.get('seed', 42))

        total_epoch = args.get('total_epoch', 2000)
        warmup_epoch = args.get('warmup_epoch', 200)
        mask_ratio = args.get('mask_ratio', 0.75)
        model_path = args.get('model_path', f'vit-t-mae-{self.run_name}.pt')

        lr_func = lambda epoch: min(
            (epoch + 1) / (warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / total_epoch * math.pi) + 1)
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_func)


        self.optim.zero_grad()
        for e in range(total_epoch):
            losses = self.train_iteration(self.steps_per_update, self.dataloader, self.device, self.model, mask_ratio)
            lr_scheduler.step()
            avg_loss = sum(losses) / len(losses)
            self.writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average training loss is {avg_loss}.')    
            self.eval_iteration(e)

            torch.save(self.model, model_path)

    def eval_iteration(self, e):
        self.model.eval()
        encoder = self.model.encoder
        decoder = self.model.decoder
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for i, (gap_slice, target, start_sec, end_sec, _, _, gap_start_sec, gap_end_sec) in enumerate(self.val_loader):
                target = target.to(self.device)
                gap_slice = gap_slice.to(self.device)

                features, backward_indexes, masked_full = encoder(gap_slice, apply_mask=False)
                predicted_val_img, mask = decoder(features, backward_indexes)
                loss = torch.mean((predicted_val_img - gap_slice) ** 2 * mask) / mask_ratio

                self.writer.add_scalar('mae_val_loss', loss.item(), global_step=e)
                self.writer.add_image('mae_original_gap', gap_slice.squeeze(0), global_step=e)
                self.writer.add_image('mae_target', target.squeeze(0), global_step=e)
                self.writer.add_image('mae_predicted', predicted_val_img.squeeze(0), global_step=e)

    def train_iteration(self, steps_per_update, dataloader, device, model, mask_ratio):
        model.train()
        losses = []
        step_count = 0
        pbar = tqdm(total=len(dataloader), desc='Training MAE', unit='batch', leave=False, dynamic_ncols=True)
        for (spectrogram_slice, start_idx, end_idx, start_sec, end_sec, _, _, start_gap_sec, end_gap_sec) in dataloader:
            step_count += 1
            spectrogram_slice = spectrogram_slice.to(device)
            predicted_spectrogram, mask = model(spectrogram_slice)
            loss = torch.mean((predicted_spectrogram - spectrogram_slice) ** 2 * mask) / mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                self.optim.step()
                self.optim.zero_grad()
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
        pbar.close()
        return losses
