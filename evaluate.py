"""
PoreSR: Evaluation Script

Runs inference on held-out test slices and computes image quality metrics
(PSNR, SSIM, MS-SSIM, LPIPS) for the seven evaluated methods: the six trained
models and the non-learned bicubic baseline.

All methods are evaluated on the same slice set. Because the 2.5D
architectures need two neighbouring slices on either side of each centre
slice, the first two and last two slices of the test partition lack a
complete neighbourhood and are excluded for every method, leaving 516 of the
520 test slices (Section 3.1).

Authors:
    Sonu Sudhikumar Seena (1), Anirban Chakraborty (2), Jingyue Hao (1), Lin Ma (1)

Implementation:
    Sonu Sudhikumar Seena

Affiliations:
    1. Department of Chemical Engineering, The University of Manchester,
       Oxford Road, Manchester M13 9PL, UK
    2. Department of Computational and Data Sciences (CDS),
       Indian Institute of Science Bangalore, Bangalore, Karnataka 560012, India

Paper:
    "Calibrated Degradation for Super-Resolution of Rock Micro-CT:
     Decoupling Image Fidelity from Petrophysical Accuracy"
    Computers & Geosciences, 2026

License: MIT
"""

import argparse
import json
import os

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import autocast
from tqdm import tqdm

from models.generator import GENERATOR_CONFIGS, build_generator


# The widest input stack across all methods is five slices, so two slices at
# each end of the test partition lack a complete neighbourhood. The same
# margin is applied to every method, including Bicubic and the 2D models, so
# that all seven are scored on an identical slice set.
EVAL_MARGIN = max(c["in_channels"] for c in GENERATOR_CONFIGS.values()) // 2

# Sliding-window reconstruction geometry, as used for the reported results.
UPSCALE = 4
PATCH_SIZE_LR = 64
PATCH_OVERLAP_LR = 16   # stride 48


def load_indices(path):
    """Load slice indices from a text file."""
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if not line.startswith("#")]


def load_slice(root, global_idx):
    """Load a single TIFF slice as float32 in [0, 1]."""
    path = os.path.join(root, f"slice_{global_idx:04d}.tif")
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        if arr.max() > 255:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
    return arr


def bicubic_upsample(lr_img, scale=4):
    """Upsample LR image using bicubic interpolation."""
    h, w = lr_img.shape
    return cv2.resize(
        lr_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
    )


def compute_metrics(sr_np, hr_np, lpips_model, device):
    """
    Compute PSNR, SSIM, MS-SSIM, and LPIPS for a single image pair.

    Parameters
    ----------
    sr_np : ndarray
        Super-resolved image, float32 in [0, 1].
    hr_np : ndarray
        Ground truth HR image, float32 in [0, 1].
    lpips_model : lpips.LPIPS
        Pre-loaded LPIPS model.
    device : torch.device
        Computation device.

    Returns
    -------
    dict
        Dictionary with psnr, ssim, ms_ssim, lpips values.
    """
    psnr_val = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    ssim_val = structural_similarity(hr_np, sr_np, data_range=1.0)

    sr_tensor = torch.from_numpy(sr_np).unsqueeze(0).unsqueeze(0)
    hr_tensor = torch.from_numpy(hr_np).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ms_ssim_val = ms_ssim(
            sr_tensor, hr_tensor, data_range=1.0, size_average=True
        ).item()

    # LPIPS expects 3-channel [-1, 1] input
    sr_rgb = sr_tensor.repeat(1, 3, 1, 1) * 2 - 1
    hr_rgb = hr_tensor.repeat(1, 3, 1, 1) * 2 - 1

    with torch.no_grad():
        lpips_val = lpips_model(sr_rgb.to(device), hr_rgb.to(device)).item()

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "ms_ssim": ms_ssim_val,
        "lpips": lpips_val,
    }


def _patch_origins(extent, patch, stride):
    """
    Sliding-window start positions covering an axis of the given extent.

    Positions run from 0 in steps of stride while a full patch fits. When the
    final position does not reach the edge, one further position flush with
    the edge is appended so that no strip is left unreconstructed. For the
    448-pixel LR slices used in the paper the loop already lands exactly on
    the edge, so no extra position is added and behaviour is unchanged.
    """
    origins = list(range(0, extent - patch + 1, stride))
    if origins and origins[-1] != extent - patch:
        origins.append(extent - patch)
    return origins or [0]


def run_inference(model, lr_stack, device, mixed_precision=True,
                  patch_size_lr=PATCH_SIZE_LR, overlap=PATCH_OVERLAP_LR):
    """
    Reconstruct a full slice from overlapping LR patches.

    The generator is applied to patch_size_lr x patch_size_lr LR patches taken
    on a stride of (patch_size_lr - overlap), each producing a 4x larger SR
    patch. Overlapping predictions are accumulated with uniform weight and
    divided by the number of contributions, which is the procedure used to
    produce the reported reconstructions. Uniform averaging is used
    deliberately; no windowed or Gaussian blending is applied.

    Parameters
    ----------
    model : SRResNet
        Trained generator.
    lr_stack : ndarray
        LR input, shape (K, H, W) for 2.5D or (1, H, W) for 2D.
    device : torch.device
        Computation device.
    mixed_precision : bool
        Whether to run the forward passes under autocast.
    patch_size_lr : int
        LR patch size in pixels. Default 64.
    overlap : int
        Overlap between adjacent LR patches in pixels. Default 16, giving a
        stride of 48.

    Returns
    -------
    ndarray
        SR output, shape (H*4, W*4), float32 in [0, 1].
    """
    _, h_lr, w_lr = lr_stack.shape
    scale = UPSCALE
    patch_size_hr = patch_size_lr * scale
    stride = patch_size_lr - overlap

    lr_tensor = torch.from_numpy(lr_stack).to(device)
    sr_full = torch.zeros(1, h_lr * scale, w_lr * scale)
    weight_map = torch.zeros(1, h_lr * scale, w_lr * scale)

    model.eval()
    with torch.no_grad():
        for i in _patch_origins(h_lr, patch_size_lr, stride):
            for j in _patch_origins(w_lr, patch_size_lr, stride):
                patch = lr_tensor[:, i:i + patch_size_lr,
                                  j:j + patch_size_lr].unsqueeze(0)
                with autocast(enabled=mixed_precision):
                    sr_patch = model(patch)
                sr_patch = sr_patch.squeeze(0).cpu().float()

                i_hr, j_hr = i * scale, j * scale
                sr_full[:, i_hr:i_hr + patch_size_hr,
                        j_hr:j_hr + patch_size_hr] += sr_patch
                weight_map[:, i_hr:i_hr + patch_size_hr,
                           j_hr:j_hr + patch_size_hr] += 1

    sr_full = sr_full / weight_map.clamp(min=1)
    return np.clip(sr_full.squeeze().numpy(), 0, 1)


def save_sr_slice(sr_np, output_dir, global_idx):
    """
    Save an SR slice as an 8-bit TIFF.

    Eight bits matches the HR and LR data and the grey-level scale on which
    the GeoDict segmentation thresholds of Section 5.2 are defined.
    """
    os.makedirs(output_dir, exist_ok=True)
    sr_uint8 = (np.clip(sr_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(sr_uint8).save(
        os.path.join(output_dir, f"slice_{global_idx:04d}.tif")
    )


def evaluate_model(model, config, test_indices, device, output_dir,
                   model_name, lpips_model):
    """
    Run inference and compute metrics for a single trained model.

    Returns a list of per-slice metric dictionaries.
    """
    # Input stack size comes from the model registry, so it can never
    # disagree with the generator that was built.
    k = GENERATOR_CONFIGS[model_name]["in_channels"]
    half_k = k // 2

    sr_dir = os.path.join(output_dir, model_name)
    os.makedirs(sr_dir, exist_ok=True)

    all_metrics = []

    for i in tqdm(range(EVAL_MARGIN, len(test_indices) - EVAL_MARGIN),
                  desc=model_name):
        centre_idx = test_indices[i]

        # Load LR stack
        lr_slices = []
        for offset in range(-half_k, half_k + 1):
            idx = test_indices[i + offset]
            lr_slices.append(
                load_slice(config["data_root_lr"], idx)
            )
        lr_stack = np.stack(lr_slices, axis=0)

        # Load HR ground truth (centre slice)
        hr_np = load_slice(config["data_root_hr"], centre_idx)

        # Reconstruct, then save as 8-bit before scoring. Metrics are
        # computed on the reopened file so that they include the same 8-bit
        # quantisation as the reported results, which were measured from the
        # saved reconstructions rather than from the float network output.
        sr_np = run_inference(model, lr_stack, device,
                              config["mixed_precision"])
        save_sr_slice(sr_np, sr_dir, centre_idx)
        sr_np = load_slice(sr_dir, centre_idx)

        metrics = compute_metrics(sr_np, hr_np, lpips_model, device)
        metrics["slice_idx"] = centre_idx
        all_metrics.append(metrics)

    return all_metrics


def evaluate_bicubic(config, test_indices, output_dir, lpips_model, device):
    """Compute metrics for the bicubic baseline."""
    sr_dir = os.path.join(output_dir, "Bicubic")
    os.makedirs(sr_dir, exist_ok=True)

    all_metrics = []

    for i in tqdm(range(EVAL_MARGIN, len(test_indices) - EVAL_MARGIN),
                  desc="Bicubic"):
        centre_idx = test_indices[i]

        lr_np = load_slice(config["data_root_lr"], centre_idx)
        hr_np = load_slice(config["data_root_hr"], centre_idx)

        sr_np = np.clip(bicubic_upsample(lr_np), 0, 1)
        save_sr_slice(sr_np, sr_dir, centre_idx)
        sr_np = load_slice(sr_dir, centre_idx)

        metrics = compute_metrics(sr_np, hr_np, lpips_model, device)
        metrics["slice_idx"] = centre_idx
        all_metrics.append(metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="PoreSR Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration JSON")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="PoreSR",
                        choices=sorted(GENERATOR_CONFIGS) + ["Bicubic"],
                        help="Model to evaluate")
    parser.add_argument("--data_splits_dir", type=str, required=True,
                        help="Directory containing test_indices.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for SR outputs and metrics")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_indices = load_indices(
        os.path.join(args.data_splits_dir, "test_indices.txt")
    )

    # Load LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)

    if args.model == "Bicubic":
        all_metrics = evaluate_bicubic(
            config, test_indices, args.output_dir, lpips_model, device
        )
    else:
        if args.checkpoint is None:
            parser.error(
                f"--checkpoint is required for {args.model}. Use "
                f"checkpoint_best.pth for the Stage 1 models, or "
                f"checkpoint_gan_best.pth for SRGAN_2D and PoreSR_GAN."
            )

        # Architecture comes from the registry, so input slice count and CBAM
        # are set independently for every model.
        model = build_generator(
            args.model,
            num_channels=config["num_channels"],
            num_blocks=config["num_residual_blocks"],
            upscale_factor=config["upscale_factor"],
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {args.model} from {args.checkpoint}")

        all_metrics = evaluate_model(
            model, config, test_indices, device, args.output_dir,
            args.model, lpips_model
        )

    # Save per-slice metrics
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(metrics_dir, f"{args.model}_per_slice.csv"),
              index=False)

    # Aggregate summary, as described in the README
    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "n_slices": int(len(df)),
    }
    for metric in ("psnr", "ssim", "ms_ssim", "lpips"):
        summary[f"{metric}_mean"] = float(df[metric].mean())
        summary[f"{metric}_std"] = float(df[metric].std())

    summary_path = os.path.join(metrics_dir, f"{args.model}_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"{args.model} Results ({len(df)} slices)")
    print(f"{'=' * 60}")
    print(f"  PSNR:    {df['psnr'].mean():.2f} +/- {df['psnr'].std():.2f} dB")
    print(f"  SSIM:    {df['ssim'].mean():.4f} +/- {df['ssim'].std():.4f}")
    print(f"  MS-SSIM: {df['ms_ssim'].mean():.4f} +/- {df['ms_ssim'].std():.4f}")
    print(f"  LPIPS:   {df['lpips'].mean():.3f} +/- {df['lpips'].std():.3f}")
    print(f"{'=' * 60}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
