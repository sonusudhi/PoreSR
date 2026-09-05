"""
PoreSR: Calibrated Degradation Pipeline

Five-component degradation pipeline empirically calibrated against a real
lower-resolution NXCT acquisition of Sherwood sandstone (10.62 um voxel,
25 mm FOV). Converts HR micro-CT slices (1792x1792, 4.73 um) to calibrated
synthetic LR images (448x448, 4x degradation).

Pipeline order:
    1. Rock-only radiometric mapping (affine, target: mu=0.269, sigma=0.012)
    2. Bias field (B-spline, amplitude capped at 0.006)
    3. PSF blur (Gaussian, sigma in [2.5, 4.0] px, 5% anisotropic)
    4. Lanczos 4x downsampling (1792x1792 -> 448x448)
    5. Poisson photon-counting noise (peak in [8263, 15345])
    6. Post-noise Gaussian smoothing (sigma=0.6)
    7. Background enforcement
    8. Ring artefacts disabled (absent at 25 mm FOV)

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

import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label as ndi_label, sum as ndi_sum
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

# Radiometric targets derived from real LR rock-only statistics
_TARGET_MEAN = 0.269
_TARGET_STD = 0.012

# Poisson peak range (recalibrated via high-pass residual matching)
_POISSON_PEAK_LOW = 8263
_POISSON_PEAK_HIGH = 15345


def load_tiff_slice(filepath):
    """Load a TIFF slice and normalise to [0, 1] float32."""
    img = Image.open(filepath).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        if arr.max() > 255:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
    return arr


def segment_rock_region(img, threshold=0.1):
    """
    Extract the largest connected component above threshold.

    The threshold of 0.1 is safe for min-max normalised images where rock
    intensity sits at approximately 0.22-0.33.
    """
    mask = img > threshold
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndi_sum(mask, labeled, range(n + 1))
    largest = sizes[1:].argmax() + 1
    return (labeled == largest).astype(bool)


def apply_radiometric_mapping(img, rock_mask, target_mean, target_std,
                              bg_value=0.0):
    """
    Affine intensity mapping on rock pixels only.

    Background is hard-set to bg_value so that no signal bleeds into the
    air region before blur and downsample operations.
    """
    out = np.full_like(img, bg_value, dtype=np.float32)
    rock = img[rock_mask]
    cur_mean = rock.mean()
    cur_std = rock.std()
    if cur_std > 1e-6:
        mapped = (rock - cur_mean) / cur_std * target_std + target_mean
    else:
        mapped = rock.copy()
    out[rock_mask] = np.clip(mapped, 0, 1)
    return out


def generate_bias_field(shape, amplitude, control_points=(4, 4)):
    """
    Generate a spatially-varying multiplicative bias field via B-spline
    interpolation, replicating beam hardening and cone-beam scatter.

    Parameters
    ----------
    shape : tuple
        Spatial dimensions (H, W).
    amplitude : float
        Peak amplitude of the bias field.
    control_points : tuple
        Number of B-spline control points in each dimension.
    """
    cp_y, cp_x = control_points
    rand_field = np.random.randn(cp_y, cp_x) * amplitude

    y_knots = np.linspace(0, shape[0] - 1, cp_y)
    x_knots = np.linspace(0, shape[1] - 1, cp_x)

    spline = RectBivariateSpline(y_knots, x_knots, rand_field, kx=3, ky=3)

    y_full = np.arange(shape[0])
    x_full = np.arange(shape[1])

    return spline(y_full, x_full)


def apply_gaussian_blur(image, sigma, anisotropic=False, angle=0.0, ratio=1.0):
    """
    Apply Gaussian PSF blur, optionally anisotropic.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Isotropic blur standard deviation in pixels.
    anisotropic : bool
        Whether to apply directional blur.
    angle : float
        Retained for interface compatibility but not applied. Anisotropic
        blur is axis-aligned, with the major axis along the row direction.
        Rotating the kernel would change the generated training data and is
        therefore not introduced here.
    ratio : float
        Anisotropy ratio sigma_major/sigma_minor (if anisotropic).
    """
    if anisotropic and ratio > 1.0:
        sigma_x = sigma
        sigma_y = sigma * ratio
        blurred = gaussian_filter(image, sigma=[sigma_y, sigma_x])
    else:
        blurred = gaussian_filter(image, sigma=sigma)
    return blurred


def downsample_4x(image):
    """Lanczos 4x downsampling via PIL. Uses uint16 intermediate to match training data generation."""
    h, w = image.shape
    img_pil = Image.fromarray((np.clip(image, 0, 1) * 65535).astype(np.uint16))
    img_down = img_pil.resize((w // 4, h // 4), Image.LANCZOS)
    return np.array(img_down, dtype=np.float32) / 65535.0


def add_poisson_noise(image, peak):
    """
    Apply Poisson photon-counting noise.

    The peak parameter controls signal-to-noise ratio: higher peak
    corresponds to lower noise.
    """
    scaled = np.clip(image * peak, 0, None)
    noisy = np.random.poisson(scaled).astype(np.float32) / peak
    return np.clip(noisy, 0, 1)


def enforce_background(img, threshold=0.02, bg_mu=0.0, bg_sigma=0.003,
                       bg_clip=0.02):
    """
    Post-downsample background re-enforcement.

    Blur and area-averaging bleed rock signal into the air boundary,
    creating a partial-volume halo. Re-segment at LR resolution and
    replace background with detector-floor noise.
    """
    lr_mask = segment_rock_region(img, threshold=threshold)
    bg = ~lr_mask
    if bg.sum() > 0:
        img[bg] = np.clip(
            np.random.normal(bg_mu, bg_sigma, bg.sum()), 0, bg_clip
        )
    return img


def sample_blur_params(calib_profile):
    """
    Sample PSF blur parameters from the calibration profile.

    The blur width is derived from the radially averaged power spectrum of
    the calibration patches rather than drawn from a single fixed range: the
    10% energy cutoff and 90% bandwidth select a base range, the mid-band
    roll-off rate then scales it, and a small normal jitter is added before
    clipping to [0.5, 4.0] px. Table 2 of the paper reports the base range
    for this instrument's calibration profile.

    These parameters are stack-consistent: all K slices in a 2.5D stack
    share the same blur, as the CT point-spread function is determined by
    detector geometry and the reconstruction kernel and does not vary
    slice-to-slice within a single scan.
    """
    blur_cfg = calib_profile["blur"]

    cutoff = np.random.uniform(*blur_cfg["cutoff_10pct_range"])
    bw = np.random.uniform(*blur_cfg["bandwidth_90_range"])
    rolloff = np.random.uniform(*blur_cfg["rolloff_rate_range"])

    freq_indicator = (cutoff + bw) / 2.0

    if freq_indicator > 0.7:
        sigma = np.random.uniform(0.5, 1.5)
    elif freq_indicator > 0.5:
        sigma = np.random.uniform(1.2, 2.5)
    elif freq_indicator > 0.3:
        sigma = np.random.uniform(2.0, 3.5)
    else:
        sigma = np.random.uniform(2.5, 4.0)

    if rolloff > 3.0:
        sigma *= 1.2
    elif rolloff < 1.5:
        sigma *= 0.8

    sigma += np.random.normal(0, 0.15)
    sigma = np.clip(sigma, 0.5, 4.0)

    aniso = np.random.rand() < blur_cfg["anisotropic_fraction_training"]
    angle = np.random.uniform(0, np.pi) if aniso else 0.0
    ratio = np.random.uniform(*blur_cfg["anisotropic_ratio_range"]) if aniso else 1.0

    return {
        "sigma": float(sigma),
        "anisotropic": aniso,
        "angle": float(angle),
        "ratio": float(ratio),
    }


def sample_noise_bias_params(calib_profile):
    """
    Sample noise and bias field parameters from the calibration profile.

    These parameters are per-slice: each slice in a 2.5D stack receives
    independent noise and bias realisations, as Poisson noise arises from
    independent photon counting statistics and bias field drift evolves
    continuously during acquisition.
    """
    bias_cfg = calib_profile["bias_field"]

    poisson_peak = np.random.uniform(_POISSON_PEAK_LOW, _POISSON_PEAK_HIGH)

    bias_amp = np.random.uniform(*bias_cfg["amplitude_range"]) * 0.5
    bias_amp = min(bias_amp, 0.5 * _TARGET_STD)  # Cap at 0.006

    return {
        "poisson_peak": float(poisson_peak),
        "bias_amp": float(bias_amp),
    }


def degrade_single_slice(hr_img, blur_params, noise_bias_params):
    """
    Apply the full calibrated degradation pipeline to one HR slice.

    Pipeline: rock mask -> radiometric mapping -> bias field -> PSF blur
    -> Lanczos 4x downsample -> Poisson noise -> background enforcement
    -> post-noise smoothing.

    Parameters
    ----------
    hr_img : ndarray
        HR slice normalised to [0, 1], shape (1792, 1792).
    blur_params : dict
        Output of sample_blur_params().
    noise_bias_params : dict
        Output of sample_noise_bias_params().

    Returns
    -------
    lr_img : ndarray
        Degraded LR slice, shape (448, 448), values in [0, 1].
    """
    img = hr_img.copy()

    # Step 1: Rock mask and radiometric mapping
    rock_mask = segment_rock_region(img, threshold=0.1)
    img = apply_radiometric_mapping(
        img, rock_mask, _TARGET_MEAN, _TARGET_STD, bg_value=0.0
    )

    # Step 2: Bias field (multiplicative, capped)
    bias_field = generate_bias_field(img.shape, noise_bias_params["bias_amp"])
    img = np.clip(img * (1.0 + bias_field), 0, 1)

    # Step 3: PSF blur
    img = apply_gaussian_blur(
        img,
        blur_params["sigma"],
        blur_params["anisotropic"],
        blur_params["angle"],
        blur_params["ratio"],
    )

    # Step 4: Lanczos 4x downsampling
    img = downsample_4x(img)

    # Step 5: Poisson photon-counting noise
    img = add_poisson_noise(img, noise_bias_params["poisson_peak"])

    # Step 6: Background enforcement
    img = enforce_background(img, threshold=0.02, bg_mu=0.0,
                             bg_sigma=0.003, bg_clip=0.02)

    # Step 7: Post-noise Gaussian smoothing (CT ramp filter apodisation)
    img = gaussian_filter(img, sigma=0.6)

    return np.clip(img, 0, 1)


def generate_synthetic_lr(hr_dir, calib_profile, output_dir, k_slices=5):
    """
    Generate calibrated synthetic LR images for all HR slices.

    Blur parameters are stack-consistent (redrawn every K slices).
    Noise and bias parameters are per-slice.

    Parameters
    ----------
    hr_dir : str
        Directory containing HR TIFF slices (slice_NNNN.tif).
    calib_profile : dict
        Calibration profile loaded from JSON.
    output_dir : str
        Output directory for synthetic LR TIFF slices.
    k_slices : int
        Stack size for blur consistency. Default: 5.
    """
    os.makedirs(output_dir, exist_ok=True)

    hr_files = sorted([
        os.path.join(hr_dir, f)
        for f in os.listdir(hr_dir)
        if f.lower().endswith((".tif", ".tiff")) and not f.startswith("._")
    ])

    total = len(hr_files)
    generated = 0
    skipped = 0

    for idx in tqdm(range(total), desc="Generating synthetic LR"):
        out_path = os.path.join(output_dir, f"slice_{idx:04d}.tif")

        if os.path.exists(out_path):
            skipped += 1
            continue

        hr_img = load_tiff_slice(hr_files[idx])

        # The global RNG is reseeded per slice so that generation is
        # deterministic and can be resumed: rerunning any slice reproduces the
        # same LR image. Blur is seeded per K-block so that a stack shares one
        # PSF; noise and bias are seeded per slice, and that seed also governs
        # the bias field, Poisson and background draws inside
        # degrade_single_slice.
        np.random.seed(42 + (idx // k_slices))
        blur_params = sample_blur_params(calib_profile)

        np.random.seed(42 + idx)
        nb_params = sample_noise_bias_params(calib_profile)

        lr_img = degrade_single_slice(hr_img, blur_params, nb_params)

        lr_uint16 = (lr_img * 65535).astype(np.uint16)
        Image.fromarray(lr_uint16).save(out_path)
        generated += 1

    print(f"Generated: {generated}, Skipped (existing): {skipped}")
    return generated


def load_calibration_profile(json_path):
    """Load a calibration profile from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)
