# PoreSR

**Calibrated Degradation for Super-Resolution of Rock Micro-CT: Decoupling Image Fidelity from Petrophysical Accuracy**

Sonu Sudhikumar Seena<sup>1</sup>, Anirban Chakraborty<sup>2</sup>, Jingyue Hao<sup>1</sup>, Lin Ma<sup>1</sup>

<sup>1</sup> Department of Chemical Engineering, The University of Manchester, Oxford Road, Manchester M13 9PL, UK
<sup>2</sup> Department of Computational and Data Sciences (CDS), Indian Institute of Science Bangalore, Bangalore, Karnataka 560012, India

Corresponding author: Lin Ma (lin.ma@manchester.ac.uk)
Code contact: Sonu Sudhikumar Seena (sonu.sudhikumarseena@postgrad.manchester.ac.uk)

Under review at *Computers & Geosciences*, 2026.

---

## Overview

PoreSR is a 2.5D residual network with CBAM attention for super-resolution of rock micro-CT images, trained on a degradation pipeline empirically calibrated against a real NXCT acquisition of Sherwood sandstone. The framework examines the relationship between image fidelity metrics and petrophysical transport properties in digital rock physics super-resolution.

This repository contains the calibrated degradation pipeline, implementations of the six learned model configurations, the training and evaluation code, and a synthetic worked example that runs end to end without the micro-CT dataset.

### What the study found

- **Calibrated degradation.** A multi-stage calibrated degradation pipeline (radiometric mapping, bias field, PSF blur, Lanczos downsampling, Poisson noise, background enforcement, post-noise smoothing) derived from a real 10.62 µm NXCT acquisition. In a matched end-to-end comparison against conventional bicubic degradation, calibration reduces the mean absolute directional permeability error of the degraded volume from 65.3% to 8.4%.

- **Volumetric context and attention are non-additive.** A controlled 2×2 factorial varies five-slice input and CBAM attention independently. Neither component alone brings the mean absolute directional permeability error below 50%; together they reduce it to 15.8%.

- **Image fidelity and transport fidelity are optimised by different architectures.** The highest-PSNR configuration in the study (SRResNet-2.5D, 39.22 dB) incurs a 51.9% mean absolute directional permeability error, while PoreSR gives up 0.21 dB and attains the lowest error of any method evaluated.

- **No single petrophysical descriptor captures transport fidelity** across the reconstructions evaluated. Distinguishing their behaviour requires joint evaluation of porosity, directional permeability, and pore-throat-size distribution.

## Repository structure

```
PoreSR/
├── configs/
│   ├── config.json                    # Training hyperparameters
│   ├── calibration_profile_25mm.json  # NXCT calibration parameters
│   └── data_splits/
│       ├── train_indices.txt          # Training split (4182 slices)
│       ├── val_indices.txt            # Validation split (519 slices)
│       └── test_indices.txt           # Test split (520 slices; 516 evaluated)
├── data/
│   └── dataset.py                     # 2D and 2.5D micro-CT data loader
├── degradation/
│   └── pipeline.py                    # Calibrated degradation pipeline
├── losses/
│   └── combined_loss.py               # L1 + MS-SSIM + gradient loss
├── models/
│   ├── generator.py                   # SRResNet backbone, CBAM, 2.5D input
│   └── discriminator.py               # PatchGAN discriminator
├── utils/
│   └── checkpoint.py                  # Checkpoint manager
├── examples/
│   └── synthetic_demo.py              # End-to-end run on generated data
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

### Computational requirements

| Requirement | Specification |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0 or later |
| CUDA | 12.1 |
| GPU memory | 16 GB minimum |
| Tested on | NVIDIA A100 (40 GB) |
| Stage 1 training | 80 000 steps per architecture |
| Stage 2 fine-tuning | 5000 steps |

The synthetic worked example in `examples/` runs on CPU and requires no GPU.

## Quick start

To verify the installation without the micro-CT dataset, run the synthetic example. It generates a small volume, applies the calibrated degradation pipeline, runs a short training loop, and reports reconstruction metrics.

```bash
python examples/synthetic_demo.py
```

Expected behaviour: the script completes in a few minutes on CPU and writes reconstructed slices and a metrics summary to `examples/output/`. It is a functional test of the pipeline, not a reproduction of the published results, which require the full HR dataset.

## Data preparation

### HR ground truth

The Sherwood sandstone micro-CT dataset is not distributed with this repository. See *Data availability* below.

Place HR slices (1792 × 1792 px, 4.73 µm voxel, 8-bit TIFF) in a directory using the naming convention `slice_0000.tif` through `slice_5232.tif`.

### Synthetic LR generation

Synthetic low-resolution training data is generated from HR by the calibrated degradation pipeline. Degradation is applied in-plane only, so slice spacing remains 4.73 µm while in-plane resolution becomes 18.92 µm.

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

The calibration profile is specific to the University of Manchester NXCT instrument and this acquisition setting. Applying the pipeline to a different instrument requires re-deriving the parameters; the procedure is described in Section 3.4 of the manuscript.

## Training

Six learned model configurations are implemented. The four factorial cells differ only in the two factors under test.

```bash
# --- 2x2 factorial ---

# SRResNet-2D: single slice, no CBAM
python train.py --config configs/config.json --model SRResNet_2D \
    --output_dir outputs/SRResNet_2D --data_splits_dir configs/data_splits

# SRResNet-2D-CBAM: single slice, CBAM
python train.py --config configs/config.json --model SRResNet_2D_CBAM \
    --output_dir outputs/SRResNet_2D_CBAM --data_splits_dir configs/data_splits

# SRResNet-2.5D: five slices, no CBAM
python train.py --config configs/config.json --model SRResNet_2_5D \
    --output_dir outputs/SRResNet_2_5D --data_splits_dir configs/data_splits

# PoreSR: five slices, CBAM
python train.py --config configs/config.json --model PoreSR \
    --output_dir outputs/PoreSR --data_splits_dir configs/data_splits

# --- Stage 2: adversarial fine-tuning ---
# These do not retrain a backbone. Each loads the selected Stage 1 checkpoint
# of its corresponding model and runs 5000 adversarial steps, so the Stage 1
# run above must be completed first.

# SRGAN-2D: fine-tunes SRResNet-2D
python train.py --config configs/config.json --model SRGAN_2D \
    --stage1_checkpoint outputs/SRResNet_2D/checkpoint_best.pth \
    --output_dir outputs/SRGAN_2D --data_splits_dir configs/data_splits

# PoreSR-GAN: fine-tunes PoreSR
python train.py --config configs/config.json --model PoreSR_GAN \
    --stage1_checkpoint outputs/PoreSR/checkpoint_best.pth \
    --output_dir outputs/PoreSR_GAN --data_splits_dir configs/data_splits
```

### Options

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to the training configuration JSON |
| `--model` | yes | One of `SRResNet_2D`, `SRResNet_2D_CBAM`, `SRResNet_2_5D`, `PoreSR`, `SRGAN_2D`, `PoreSR_GAN` |
| `--output_dir` | yes | Directory for checkpoints and training logs |
| `--data_splits_dir` | yes | Directory containing the three index files |
| `--stage1_checkpoint` | GAN only | Stage 1 checkpoint to fine-tune from. Required for `SRGAN_2D` and `PoreSR_GAN` |

**Outputs.** Stage 1 runs write `checkpoint_best.pth`, selected by best validation MS-SSIM, plus periodic checkpoints and a training log. Stage 2 runs write `checkpoint_gan_best.pth`, selected by the same criterion, plus a final-step snapshot `generator_gan_step_5000.pth`. Use the `_best` files for evaluation.

### Training configuration

| Parameter | Value |
|---|---|
| Stage 1 steps | 80 000 |
| Stage 2 steps | 5000 |
| Batch size | 16 (A100) |
| Patch size | 64 × 64 (LR) / 256 × 256 (HR) |
| Learning rate | 2e-4, warmup 5000 steps, cosine to 5e-5 |
| Loss weights | L1 1.0, MS-SSIM 0.1, gradient 0.01 |
| Adversarial weight | 0.001 |
| Stage 2 generator / discriminator LR | 1e-5 / 4e-5 |

The four factorial cells use the same fixed data partitions and calibrated synthetic LR dataset, loss, optimiser schedule, and backbone. CBAM contributes 9760 parameters and the five-slice input 20 736, so trainable parameter count varies by 2.0% across the four cells. The PoreSR generator contains 1 554 996 trainable parameters.

## Evaluation

```bash
# Stage 1 models
python evaluate.py --config configs/config.json --model PoreSR \
    --checkpoint outputs/PoreSR/checkpoint_best.pth \
    --data_splits_dir configs/data_splits --output_dir results/

# Non-learned bicubic baseline (no checkpoint required)
python evaluate.py --config configs/config.json --model Bicubic \
    --data_splits_dir configs/data_splits --output_dir results/
```

**Inputs.** A trained checkpoint (except for `Bicubic`), the synthetic LR test slices, and the corresponding HR slices.

**Outputs.** Reconstructed 8-bit TIFF slices for the 516-slice test set, a per-slice CSV, and `<model>_metrics_summary.json` giving the model, checkpoint, slice count, and mean and standard deviation of PSNR, SSIM, MS-SSIM and LPIPS.

Full slices are reconstructed from overlapping 64 x 64 LR patches on a stride of 48, with overlapping predictions averaged uniformly. Metrics are computed on the saved 8-bit reconstructions rather than on the float network output, so that they include the same quantisation as the reported results.

Petrophysical evaluation (porosity, Stokes–Brinkman permeability, pore-throat-size distribution) was performed in GeoDict, which is commercial software and not included here. Reconstructed volumes written by `evaluate.py` are the input to that stage. The same GeoDict hysteresis threshold-selection protocol was applied to every volume: the automatic estimate minus 1.0 grey-level unit. The resulting numerical operating threshold was image-dependent.

## Results

Evaluated on 516 held-out test slices. HR reference permeabilities are *K*x = 459.72, *K*y = 518.60, *K*z = 181.30 mD (1 mD = 9.869 × 10⁻¹⁶ m²).

| Method | PSNR (dB) | *K*x (mD) | *K*y (mD) | *K*z (mD) | Mean abs. directional error (%) |
|---|---|---|---|---|---|
| HR reference | — | 459.72 | 518.60 | 181.30 | — |
| Bicubic | 34.69 | 353.62 | 385.98 | 104.80 | 30.3 |
| SRResNet-2D | 37.70 | 249.72 | 282.35 | 28.54 | 58.5 |
| SRGAN-2D | 37.89 | 311.12 | 338.55 | 39.27 | 48.5 |
| SRResNet-2D-CBAM | 38.22 | 272.17 | 293.87 | 34.83 | 55.0 |
| SRResNet-2.5D | **39.22** | 759.96 | 857.15 | 226.95 | 51.9 |
| **PoreSR** | 39.01 | 491.36 | 552.39 | 119.79 | **15.8** |
| PoreSR-GAN | 38.66 | 411.83 | 470.26 | 94.68 | 22.5 |

Mean absolute directional error is the equally weighted mean of absolute relative errors in *K*x, *K*y and *K*z. It is a descriptive summary of three-direction transport accuracy, not a petrophysical property.

No method achieves both the highest image fidelity and the lowest permeability error. SRResNet-2.5D attains the highest PSNR while ranking fifth of seven on permeability accuracy, and is the only reconstruction to overestimate permeability in all three directions.

## Experimental design

Seven reconstruction methods: six trained models and one non-learned baseline.

### 2×2 factorial

Mean absolute directional permeability error, with the two factors varied independently:

| | No CBAM | CBAM |
|---|---|---|
| **1 slice (2D)** | SRResNet-2D — 58.5% | SRResNet-2D-CBAM — 55.0% |
| **5 slices (2.5D)** | SRResNet-2.5D — 51.9% | PoreSR — **15.8%** |

Estimating the interaction as the difference between the effect of adding CBAM in the 2.5D configuration and in the 2D configuration gives −63.3, −61.0 and −62.6 percentage points in X, Y and Z. These are descriptive point estimates from single training runs rather than formally estimated factorial effects.

### Other methods

- **Bicubic** — non-learned baseline.
- **SRGAN-2D** — adversarial fine-tuning of SRResNet-2D.
- **PoreSR-GAN** — adversarial fine-tuning of PoreSR. Included to quantify the adversarial response, not proposed as a recommended method for petrophysical applications.

All learned configurations use the same calibrated degradation dataset. The Stage 1 models share the common reconstruction-training configuration, while the adversarial variants additionally use the Stage 2 settings listed above.

## Data availability

The Sherwood sandstone micro-CT dataset was acquired at the University of Manchester NXCT facility and is not distributed with this repository. Access enquiries should be directed to the corresponding author.

Trained model weights are not distributed either. This repository provides the implementation and the training and evaluation code; reproducing the reported models requires the HR dataset and the training procedure described above.

The synthetic worked example in `examples/` allows the degradation pipeline, model definitions, training loop, and evaluation workflow to be exercised end to end without it.

## Limitations

Validation is internal to a single Sherwood sandstone specimen. No cross-sample, cross-instrument, or cross-lithology testing was performed. Calibration parameters are specific to the University of Manchester NXCT instrument and this acquisition setting, and residual mismatches in noise amplitude and low-frequency bias remain. Permeability was computed with the LIR solver alone and without comparison against laboratory measurement. Each architecture is a single trained model without seed replication. See Section 6.6 of the manuscript for the full statement.

## Citation

```bibtex
@article{seena2026poresr,
  title   = {Calibrated Degradation for Super-Resolution of Rock Micro-CT:
             Decoupling Image Fidelity from Petrophysical Accuracy},
  author  = {Sudhikumar Seena, Sonu and Chakraborty, Anirban and
             Hao, Jingyue and Ma, Lin},
  journal = {Computers \& Geosciences},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
