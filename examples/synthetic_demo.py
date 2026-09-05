"""
PoreSR: Synthetic End-to-End Demo

Runs the complete PoreSR workflow on a small synthetic volume so that the
degradation pipeline, dataset, generator, loss and evaluation code can be
exercised without the Sherwood sandstone micro-CT dataset, which is not
distributed with this repository.

The demo is a functional test, not a reproduction of the published results.
It trains for a few hundred steps on a 40-slice synthetic volume, whereas the
reported models were trained for 80 000 steps on 4182 real HR slices. The
reconstruction quality it reaches is not meaningful; what it demonstrates is
that every stage of the workflow runs and produces output of the expected
shape and dynamic range.

Usage
-----
Run from the repository root:

    python examples/synthetic_demo.py

Options:

    --steps N        training steps (default 100)
    --slices N       synthetic volume depth (default 40)
    --size N         synthetic HR slice size in pixels (default 512)
    --device DEV     'cpu' or 'cuda' (default: cuda if available)
    --output DIR     output directory (default examples/output)

Outputs, written to the output directory:

    hr/              synthetic HR slices, 16-bit TIFF
    lr/              calibrated synthetic LR slices, 16-bit TIFF
    splits/          train, validation and test index files
    comparison.png   HR, LR, bicubic and PoreSR for one test slice
    summary.txt      PSNR and SSIM for bicubic and PoreSR

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
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

# Allow the demo to be run from the repository root without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import MicroCTDataset                      # noqa: E402
from degradation.pipeline import (                           # noqa: E402
    degrade_single_slice,
    load_calibration_profile,
    sample_blur_params,
    sample_noise_bias_params,
)
from losses.combined_loss import CombinedLoss                # noqa: E402
from models.generator import build_generator, count_parameters  # noqa: E402

CALIB_PATH = REPO_ROOT / "configs" / "calibration_profile_25mm.json"

# Patch geometry, matching the published configuration.
PATCH_HR = 256
UPSCALE = 4
K_SLICES = 5


def make_synthetic_volume(n_slices, size, seed=0):
    """
    Generate a synthetic sandstone-like volume.

    A smoothed three-dimensional random field is thresholded into grain and
    pore phases, then masked to a circular core so that the volume has a
    background region below the rock-segmentation threshold used by the
    degradation pipeline. Smoothing along the third axis gives the inter-slice
    continuity that the 2.5D architectures exploit.

    Returns
    -------
    ndarray
        Volume of shape (n_slices, size, size), float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n_slices, size, size)).astype(np.float32)

    # Anisotropic smoothing: coarser in plane than through plane, so that
    # features persist across several slices.
    field = gaussian_filter(field, sigma=(1.5, 4.0, 4.0))
    field = (field - field.mean()) / (field.std() + 1e-8)

    # Threshold into two phases at roughly 15% pore fraction.
    pore = field < np.percentile(field, 15.0)
    volume = np.where(pore, 0.24, 0.56).astype(np.float32)

    # Soften phase boundaries so the image is not perfectly binary.
    volume = gaussian_filter(volume, sigma=(0.0, 0.8, 0.8))

    # Circular core mask; outside the core is air, below the 0.1 rock threshold.
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2.0
    radius = size * 0.46
    core = ((yy - centre) ** 2 + (xx - centre) ** 2) <= radius ** 2
    volume = volume * core[None, :, :]

    return np.clip(volume, 0.0, 1.0)


def save_slice(arr, path):
    """
    Save a float array in [0, 1] as an 8-bit TIFF.

    Eight bits is used deliberately. The loaders in data/dataset.py and
    evaluate.py read slices with PIL convert("L"), which clips 16-bit values
    at 255 rather than rescaling them, so a 16-bit file would be read back as
    a saturated binary image.
    """
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def write_hr_slices(volume, hr_dir):
    """Write the synthetic HR volume to disk in the expected naming scheme."""
    hr_dir.mkdir(parents=True, exist_ok=True)
    for i, sl in enumerate(volume):
        save_slice(sl, hr_dir / f"slice_{i:04d}.tif")


def degrade_volume(volume, lr_dir, calib):
    """
    Apply the calibrated degradation pipeline to every slice.

    Blur parameters are held constant within non-overlapping blocks of
    K_SLICES consecutive slices and resampled between blocks; noise and bias
    parameters are drawn per slice. This reproduces the sampling scheme used
    to generate the training data (Section 3.3).
    """
    lr_dir.mkdir(parents=True, exist_ok=True)
    for i, sl in enumerate(volume):
        np.random.seed(42 + (i // K_SLICES))
        blur = sample_blur_params(calib)
        np.random.seed(42 + i)
        noise_bias = sample_noise_bias_params(calib)
        lr = degrade_single_slice(sl, blur, noise_bias)
        save_slice(lr, lr_dir / f"slice_{i:04d}.tif")


def write_splits(n_slices, splits_dir):
    """
    Write contiguous train, validation and test index files.

    Each split must hold at least K_SLICES slices so that the 2.5D dataset has
    at least one valid centre slice.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    n_test = max(K_SLICES + 3, n_slices // 5)
    n_val = max(K_SLICES + 3, n_slices // 5)
    n_train = n_slices - n_val - n_test
    if n_train < K_SLICES + 3:
        raise ValueError(
            f"--slices {n_slices} is too small; use at least {3 * (K_SLICES + 3)}"
        )

    bounds = {
        "train": range(0, n_train),
        "val": range(n_train, n_train + n_val),
        "test": range(n_train + n_val, n_slices),
    }
    for name, rng_ in bounds.items():
        with open(splits_dir / f"{name}_indices.txt", "w") as f:
            f.write("\n".join(str(i) for i in rng_) + "\n")
    return {k: list(v) for k, v in bounds.items()}


def bicubic_upsample(lr, scale=UPSCALE):
    """Upsample an LR slice by bicubic interpolation, for the baseline."""
    h, w = lr.shape
    img = Image.fromarray((np.clip(lr, 0, 1) * 255).astype(np.uint8))
    up = img.resize((w * scale, h * scale), Image.BICUBIC)
    return np.array(up, dtype=np.float32) / 255.0


def load_float(path):
    """Load a TIFF slice as float32 in [0, 1], matching data/dataset.py."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def train_briefly(model, loader, device, steps, seed=42):
    """Run a short reconstruction-only training loop and return mean loss."""
    torch.manual_seed(seed)
    model = model.to(device).train()
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

    losses = []
    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        lr_imgs = batch["lr"].to(device)
        hr_imgs = batch["hr"].to(device)

        optimizer.zero_grad()
        loss, _ = criterion(model(lr_imgs), hr_imgs)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            recent = np.mean(losses[-10:])
            print(f"    step {step + 1:4d}/{steps}  loss {recent:.4f}")
    return float(np.mean(losses[-10:]))


def reconstruct(model, lr_stack, device):
    """Reconstruct the centre slice of one LR stack."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(lr_stack).unsqueeze(0).to(device)
        sr = model(x).squeeze().cpu().float().numpy()
    return np.clip(sr, 0.0, 1.0)


def save_comparison(hr, lr, bic, sr, path):
    """Write a four-panel comparison strip as a PNG."""
    lr_display = np.array(
        Image.fromarray((np.clip(lr, 0, 1) * 255).astype(np.uint8)).resize(
            (hr.shape[1], hr.shape[0]), Image.NEAREST
        ),
        dtype=np.float32,
    ) / 255.0
    strip = np.concatenate([hr, lr_display, bic, sr], axis=1)
    Image.fromarray((np.clip(strip, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="PoreSR synthetic end-to-end demo. Runs the full workflow "
                    "on generated data; not a reproduction of published results."
    )
    parser.add_argument("--steps", type=int, default=100,
                        help="Training steps (default: 100)")
    parser.add_argument("--slices", type=int, default=40,
                        help="Synthetic volume depth (default: 40)")
    parser.add_argument("--size", type=int, default=512,
                        help="Synthetic HR slice size in pixels (default: 512)")
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu' or 'cuda' (default: cuda if available)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent / "output"),
                        help="Output directory")
    args = parser.parse_args()

    if args.size % (UPSCALE * 4) != 0 or args.size < PATCH_HR:
        parser.error(
            f"--size must be a multiple of {UPSCALE * 4} and at least {PATCH_HR}"
        )

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PoreSR synthetic demo — functional test, not a reproduction")
    print("=" * 68)
    print(f"  device        {device}")
    print(f"  volume        {args.slices} slices of {args.size} x {args.size}")
    print(f"  training      {args.steps} steps "
          f"(published models: 80 000 steps on 4182 real slices)")
    print(f"  output        {out}")

    if not CALIB_PATH.exists():
        sys.exit(f"Calibration profile not found: {CALIB_PATH}")
    calib = load_calibration_profile(str(CALIB_PATH))

    print("\n[1/5] Generating synthetic HR volume")
    volume = make_synthetic_volume(args.slices, args.size)
    hr_dir = out / "hr"
    write_hr_slices(volume, hr_dir)
    print(f"      wrote {args.slices} slices to {hr_dir}")

    print("\n[2/5] Applying calibrated degradation pipeline")
    lr_dir = out / "lr"
    degrade_volume(volume, lr_dir, calib)
    lr_size = args.size // UPSCALE
    print(f"      wrote {args.slices} slices of {lr_size} x {lr_size} "
          f"to {lr_dir}")

    print("\n[3/5] Building splits and dataset")
    splits = write_splits(args.slices, out / "splits")
    train_ds = MicroCTDataset(
        slice_indices=splits["train"],
        data_root_hr=str(hr_dir), data_root_lr=str(lr_dir),
        k=K_SLICES, patch_size_hr=PATCH_HR, patches_per_image=4, phase="train",
    )
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    print(f"      train {len(splits['train'])} / val {len(splits['val'])} / "
          f"test {len(splits['test'])} slices, {len(train_ds)} training patches")

    print("\n[4/5] Training PoreSR")
    model = build_generator("PoreSR")
    print(f"      {count_parameters(model):,} trainable parameters")
    final_loss = train_briefly(model, loader, device, args.steps)

    print("\n[5/5] Evaluating on the test split")
    half_k = K_SLICES // 2
    centres = splits["test"][half_k:len(splits["test"]) - half_k]
    rows = []
    first = None
    for centre in centres:
        stack = np.stack(
            [load_float(lr_dir / f"slice_{centre + o:04d}.tif")
             for o in range(-half_k, half_k + 1)], axis=0)
        hr = load_float(hr_dir / f"slice_{centre:04d}.tif")
        sr = reconstruct(model, stack, device)
        bic = bicubic_upsample(stack[half_k])
        rows.append((
            peak_signal_noise_ratio(hr, bic, data_range=1.0),
            structural_similarity(hr, bic, data_range=1.0),
            peak_signal_noise_ratio(hr, sr, data_range=1.0),
            structural_similarity(hr, sr, data_range=1.0),
        ))
        if first is None:
            first = (hr, stack[half_k], bic, sr)

    m = np.mean(np.array(rows), axis=0)
    save_comparison(*first, out / "comparison.png")

    summary = (
        "PoreSR synthetic demo\n"
        "Functional test only. Not a reproduction of the published results.\n\n"
        f"volume            {args.slices} slices of {args.size} x {args.size}\n"
        f"training steps    {args.steps}\n"
        f"test slices       {len(centres)}\n"
        f"final loss        {final_loss:.4f}\n\n"
        f"{'method':10}{'PSNR (dB)':>12}{'SSIM':>10}\n"
        f"{'bicubic':10}{m[0]:>12.2f}{m[1]:>10.4f}\n"
        f"{'PoreSR':10}{m[2]:>12.2f}{m[3]:>10.4f}\n"
    )
    (out / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"Comparison image: {out / 'comparison.png'}")
    print("Panels, left to right: HR, LR (nearest), bicubic, PoreSR")
    print("\nDemo complete. All pipeline stages ran successfully.")


if __name__ == "__main__":
    main()
