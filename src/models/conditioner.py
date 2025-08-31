import typing as tp
from torch import nn
import torch
import gc
import logging
import warnings
import numpy as np
from transformers import ClapModel, ClapProcessor
import laion_clap

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
    ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()
        self.finetune = False

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


def clap_load_state_dict(clap_ckpt_path, clap_model):
    state_dict = torch.load(clap_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    # Remove "module." from state dict keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Fix for transformers library
    removed_keys = ["text_branch.embeddings.position_ids"]
    for removed_key in removed_keys:
        if removed_key in state_dict:
            del state_dict[removed_key]

    clap_model.load_state_dict(state_dict, strict=False)
class CLAPAudioConditioner(Conditioner):
    def __init__(self,
                 output_dim: int,
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base",
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # Allowlist NumPy scalar needed by older CLAP checkpoints (PyTorch 2.6 sets weights_only=True by default)
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:

                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type)
                clap_load_state_dict(clap_ckpt_path, model.model)
                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CLAP audio embeddings and project to the requested output_dim.

        Expects x with shape (B, 1, T) or (B, T). Returns (B, output_dim).
        """
        if x.dim() == 3:
            # squeeze channel dim (mono)
            x = x.squeeze(1)
        # Use laion_clap's tensor path so gradients are preserved if caller enables them
        emb = self.model.get_audio_embedding_from_data(x, use_tensor=True)
        return self.proj_out(emb)


if __name__ == "__main__":
    conditioner = CLAPAudioConditioner(
        1024,
        "pretrained/music_audioset_epoch_15_esc_90.14.pt",
        audio_model_type="HTSAT-base",
        enable_fusion=True,
        project_out=True,
    )
    x = torch.randn(2, 1, 16000 * 5)
    with torch.no_grad():
        y = conditioner(x)
    print(y.shape)