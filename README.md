# Audio Gap Filling

**Audio Gap Filling** is a research framework for experimenting with audio inpainting.  
It provides reusable building blocks for training and evaluating different generative approaches on masked or corrupted audio segments.


## Reconstruction Strategies

The current focus is on two reconstruction strategies:

- **MAE (Masked Autoencoder) reconstruction** – operates on mel-spectrogram patches and rebuilds masked regions using a Vision Transformer architecture.  
- **Stable diffusion reconstruction** – inspired by *Stable Audio Open 2.0*, with an intermediate VAE for waveform-to-latent mapping. Diffusion training on the learned latent space is planned but not yet implemented.

> Early experiments implementing the method from *Moliner2024 – Diffusion-based audio in-painting*  
> (`src/models/unet_cqt_oct_with_projattention_adaLN_2.py`,  
> `src/training/diff_trainer.py`, `src/training/edm.py`) remain in the repository for reference,  
> but are incomplete and superseded by the new Stable Audio–inspired direction.

---

## Project Highlights

- 🏗 **Factory-based training pipeline** – instantiate models, datasets, and trainers purely from YAML configuration.  
- 🎛 **MAE Vision Transformer** – custom masked autoencoder for spectrogram patches, optional use of a pre-trained AudioMAE backbone.  
- 🎵 **VAE for waveform compression** – encoder/decoder network producing compact latent representations; intended as the foundation for latent diffusion.  
- 📂 **Configurable datasets** – mel-spectrogram slices, raw audio folders, or specialized waveform gap datasets.  
- ⚙️ **Training utilities** – logging via TensorBoard, metric helpers, gradient accumulation, perceptual losses (LPIPS), and reproducible seeding.  

---

## Directory Structure
```
audio_gap_filling/
├── configs/              # YAML files controlling data, model, and training options
├── src/
│   ├── config/           # Configuration loader & helpers
│   ├── core/             # Base classes for datasets, models, trainers
│   ├── data/             # Dataset implementations (mel, raw audio, VAE/gap datasets)
│   ├── models/           # MAE ViT, VAE, diffusion UNet (legacy), utilities
│   ├── training/         # Trainer classes (MAETrainer, VAETrainer, legacy DiffusionTrainer)
│   ├── utils/            # Metrics, math helpers, augmentation utilities
│   └── factory.py        # Orchestrates model/dataset/trainer creation & pipeline
├── examples/             # Minimal example showing custom model integration
├── train.py              # Command-line entry point for training
└── requirements*.txt     # Dependency lists

```
## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```


### 2. Prepare your dataset
- Place audio files in the folder expected by the chosen dataset class.
- Adjust dataset paths in your configuration file.

### 3. Choose or create a configuration
- Configuration examples live in configs/.
- Each file defines the dataset type, model architecture, trainer, optimization parameters, and logging paths.
Example: mask050_l1_01.yaml trains the MAE with 50 % masking and an L1 loss component.

### 4. Run training
```python
python train.py --config mask050_l1_01
# or with an explicit file path:
python train.py --config configs/mask050_l1_01.yaml
```
TODO

- Implement training loop for Stable Audio–style diffusion.

- Define the UNet-based diffusion model (instead of a DiT).

- Add EDM helper functions (src/training/edm.py) adapted for waveform-latent diffusion.

- Integrate new UNet diffusion into the factory-based pipeline (configs + trainer).

Already done: conditioner.py we'll use to condition the learning of UNET witt sorrounding context of the gap