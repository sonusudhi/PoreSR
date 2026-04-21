# PoreSR

**Calibrated Degradation for Super-Resolution of Rock Micro-CT: Decoupling Image Fidelity from Petrophysical Accuracy**

Sonu Sudhikumar Seena<sup>1</sup>, Anirban Chakraborty<sup>2</sup>, Jingyue Hao<sup>1</sup>, Lin Ma<sup>1</sup>

<sup>1</sup> Department of Chemical Engineering, The University of Manchester, UK  
<sup>2</sup> Department of Computational and Data Sciences (CDS), Indian Institute of Science Bangalore, India

Corresponding author: Sonu Sudhikumar Seena (sonu.sudhikumarseena@postgrad.manchester.ac.uk)

Submitted to *Computers & Geosciences*, 2026.

---

## Overview

PoreSR is a 2.5D residual network with CBAM attention for super-resolution of rock micro-CT images, trained on a calibrated degradation pipeline empirically derived from a real NXCT acquisition of Sherwood sandstone. The framework addresses the disconnect between image fidelity metrics and petrophysical transport properties in digital rock physics SR.

Key contributions:

- **Calibrated degradation pipeline**: Five-component pipeline (radiometric mapping, bias field, PSF blur, Lanczos downsampling, Poisson noise) calibrated against real 10.6 µm NXCT acquisition statistics.
- **2.5D volumetric SR architecture**: Five-slice input stacks with CBAM attention preserve inter-slice pore connectivity, achieving a 4.3x permeability improvement over equivalent 2D architectures.
- **Three-metric petrophysical validation**: Porosity, Stokes-flow permeability, and pore throat size distribution (D10, D50, D90) proposed as the minimum evaluation standard for digital rock physics SR.

## Repository Structure

```
PoreSR/
├── configs/
│   ├── config.json                    # Training hyperparameters
│   ├── calibration_profile_25mm.json  # NXCT calibration parameters
│   └── data_splits/
│       ├── train_indices.txt          # 80% training split (4,182 slices)
│       ├── val_indices.txt            # 10% validation split (519 slices)
│       └── test_indices.txt           # 10% test split (520 slices)
├── data/
│   └── dataset.py                     # 2D and 2.5D micro-CT data loader
├── degradation/
│   └── pipeline.py                    # Calibrated degradation pipeline
├── losses/
│   └── combined_loss.py               # L1 + MS-SSIM + Gradient loss
├── models/
│   ├── generator.py                   # SRResNet + CBAM generator
│   └── discriminator.py               # PatchGAN discriminator
├── utils/
│   └── checkpoint.py                  # Checkpoint manager
├── train.py                           # Training script (Stage 1 + Stage 2)
├── evaluate.py                        # Inference and metrics computation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

## Installation

```bash
git clone https://github.com/sonusudhi/PoreSR.git
cd PoreSR
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1 on NVIDIA A100 (40 GB).

## Data Preparation

### HR Ground Truth

Place HR Sherwood sandstone micro-CT slices (1792 x 1792 px, 4.73 µm voxel, 16-bit TIFF) in a directory with naming convention `slice_0000.tif` through `slice_5232.tif`.

### Synthetic LR Generation

Generate calibrated synthetic LR images using the degradation pipeline:

```python
from degradation.pipeline import generate_synthetic_lr, load_calibration_profile

calib = load_calibration_profile("configs/calibration_profile_25mm.json")

generate_synthetic_lr(
    hr_dir="/path/to/HR_Sandstone",
    calib_profile=calib,
    output_dir="/path/to/LR_Synthetic_Sandstone",
    k_slices=5,
)
```

## Training

Train any of the four model variants:

```bash
# PoreSR (2.5D + CBAM, Stage 1 only)
python train.py \
    --config configs/config.json \
    --model PoreSR \
    --output_dir outputs/PoreSR \
    --data_splits_dir configs/data_splits

# PoreSR-GAN (2.5D + CBAM, Stage 1 + Stage 2)
python train.py \
    --config configs/config.json \
    --model PoreSR_GAN \
    --output_dir outputs/PoreSR_GAN \
    --data_splits_dir configs/data_splits

# SRResNet-2D (ablation baseline)
python train.py \
    --config configs/config.json \
    --model SRResNet_2D \
    --output_dir outputs/SRResNet_2D \
    --data_splits_dir configs/data_splits

# SRGAN-2D (ablation baseline)
python train.py \
    --config configs/config.json \
    --model SRGAN_2D \
    --output_dir outputs/SRGAN_2D \
    --data_splits_dir configs/data_splits
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Total steps (Stage 1) | 80,000 |
| GAN steps (Stage 2) | 5,000 |
| Batch size | 16 (A100) |
| Patch size | 64 x 64 (LR) / 256 x 256 (HR) |
| Learning rate | 2e-4 (warmup 5,000 steps, cosine to 5e-5) |
| Loss weights | L1: 1.0, MS-SSIM: 0.1, Gradient: 0.01 |
| GAN adversarial weight | 0.001 |

## Evaluation

```bash
# Evaluate PoreSR
python evaluate.py \
    --config configs/config.json \
    --model PoreSR \
    --checkpoint outputs/PoreSR/checkpoint_best.pth \
    --data_splits_dir configs/data_splits \
    --output_dir results/

# Evaluate bicubic baseline
python evaluate.py \
    --config configs/config.json \
    --model Bicubic \
    --data_splits_dir configs/data_splits \
    --output_dir results/
```

## Results

Evaluated on 516 held-out test slices with GeoDict petrophysical simulation:

| Method | PSNR (dB) | K_z (mDarcy) | K Error (%) |
|--------|-----------|--------------|-------------|
| HR Ground Truth | -- | 132.72 | -- |
| Bicubic | 34.69 | 81.28 | -38.8 |
| SRResNet-2D | 37.70 | 22.30 | -83.2 |
| SRGAN-2D | 37.89 | 30.73 | -76.9 |
| **PoreSR** | **39.01** | **95.54** | **-28.0** |
| PoreSR-GAN | 38.66 | 72.20 | -45.6 |

PoreSR is the only method to achieve joint optimality in both image quality and petrophysical accuracy.

## Ablation Design

The five-method comparison is structured as a controlled ablation:

- **Bicubic**: Non-learned baseline.
- **SRResNet-2D**: 2D backbone, no CBAM, reconstruction loss only.
- **SRGAN-2D**: 2D backbone, no CBAM, + adversarial fine-tuning.
- **PoreSR**: 2.5D (K=5), CBAM attention, reconstruction loss only.
- **PoreSR-GAN**: 2.5D (K=5), CBAM attention, + adversarial fine-tuning.

All trained models use identical calibrated degradation data and hyperparameters.

## Citation

If you use this code, please cite:

```bibtex
@article{sudhikumarseena2026poresr,
  title={Calibrated Degradation for Super-Resolution of Rock Micro-CT: 
         Decoupling Image Fidelity from Petrophysical Accuracy},
  author={Sudhikumar Seena, Sonu and Ma, Lin and Chakraborty, Anirban and Hao, Jingyue},
  journal={Computers \& Geosciences},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
