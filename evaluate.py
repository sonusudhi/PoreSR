"""
PoreSR: Evaluation Script

Runs inference on held-out test slices and computes image quality metrics
(PSNR, SSIM, MS-SSIM, LPIPS) for all SR methods including bicubic baseline.

Authors:
    Sonu Sudhikumar Seena (1), Anirban Chakraborty (2), Jingyue Hao (1), Lin Ma (1)

Implementation:
    Sonu Sudhikumar Seena

Affiliations:
    1. Department of Chemical Engineering, The University of Manchester, UK
    2. Department of Computational and Data Sciences (CDS),
       Indian Institute of Science Bangalore, India

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

from models.generator import SRResNet


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


def run_inference(model, lr_stack, device, mixed_precision=True):
    """
    Run SR inference on a single LR input stack.

    Parameters
    ----------
    model : SRResNet
        Trained generator model.
    lr_stack : ndarray
        LR input, shape (K, H, W) for 2.5D or (1, H, W) for 2D.
    device : torch.device
        Computation device.

    Returns
    -------
    ndarray
        SR output, shape (H*4, W*4), float32 in [0, 1].
    """
    lr_tensor = torch.from_numpy(lr_stack).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with autocast(enabled=mixed_precision):
            sr_tensor = model(lr_tensor)

    sr_np = sr_tensor.squeeze().cpu().float().numpy()
    return np.clip(sr_np, 0, 1)


def save_sr_slice(sr_np, output_dir, global_idx):
    """Save an SR slice as 16-bit TIFF."""
    os.makedirs(output_dir, exist_ok=True)
    sr_uint16 = (sr_np * 65535).astype(np.uint16)
    Image.fromarray(sr_uint16).save(
        os.path.join(output_dir, f"slice_{global_idx:04d}.tif")
    )


def evaluate_model(model, config, test_indices, device, output_dir,
                   model_name, lpips_model):
    """
    Run inference and compute metrics for a single trained model.

    Returns a list of per-slice metric dictionaries.
    """
    k = config["k_slices_25d"] if "2.5D" in model_name or "PoreSR" in model_name \
        else config["k_slices_2d"]
    half_k = k // 2

    sr_dir = os.path.join(output_dir, model_name)
    os.makedirs(sr_dir, exist_ok=True)

    all_metrics = []

    for i in tqdm(range(half_k, len(test_indices) - half_k),
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

        # Inference on full LR slice
        sr_np = run_inference(model, lr_stack, device)
        save_sr_slice(sr_np, sr_dir, centre_idx)

        # Metrics on full slice
        metrics = compute_metrics(sr_np, hr_np, lpips_model, device)
        metrics["slice_idx"] = centre_idx
        all_metrics.append(metrics)

    return all_metrics


def evaluate_bicubic(config, test_indices, output_dir, lpips_model, device):
    """Compute metrics for the bicubic baseline."""
    half_k = config["k_slices_25d"] // 2
    sr_dir = os.path.join(output_dir, "Bicubic")
    os.makedirs(sr_dir, exist_ok=True)

    all_metrics = []

    for i in tqdm(range(half_k, len(test_indices) - half_k), desc="Bicubic"):
        centre_idx = test_indices[i]

        lr_np = load_slice(config["data_root_lr"], centre_idx)
        hr_np = load_slice(config["data_root_hr"], centre_idx)

        sr_np = bicubic_upsample(lr_np)
        sr_np = np.clip(sr_np, 0, 1)
        save_sr_slice(sr_np, sr_dir, centre_idx)

        # Metrics on full slice
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
                        choices=["SRResNet_2D", "SRGAN_2D",
                                 "PoreSR", "PoreSR_GAN", "Bicubic"],
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
        # Determine architecture
        if args.model in ("SRResNet_2D", "SRGAN_2D"):
            in_channels = config["k_slices_2d"]
            use_cbam = False
        else:
            in_channels = config["k_slices_25d"]
            use_cbam = True

        model = SRResNet(
            in_channels=in_channels,
            num_channels=config["num_channels"],
            num_blocks=config["num_residual_blocks"],
            upscale_factor=config["upscale_factor"],
            use_cbam=use_cbam,
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

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"{args.model} Results ({len(df)} slices)")
    print(f"{'=' * 60}")
    print(f"  PSNR:    {df['psnr'].mean():.2f} +/- {df['psnr'].std():.2f} dB")
    print(f"  SSIM:    {df['ssim'].mean():.4f} +/- {df['ssim'].std():.4f}")
    print(f"  MS-SSIM: {df['ms_ssim'].mean():.4f} +/- {df['ms_ssim'].std():.4f}")
    print(f"  LPIPS:   {df['lpips'].mean():.3f} +/- {df['lpips'].std():.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
