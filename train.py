"""
PoreSR: Training Script

Two-stage training for the seven-method comparison:
    Stage 1: Reconstruction (L1 + MS-SSIM + gradient loss), 80 000 steps
    Stage 2: Adversarial fine-tuning (PatchGAN), 5000 steps

The two stages are run as separate commands. Stage 2 does not retrain a
backbone: SRGAN_2D and PoreSR_GAN are initialised from the selected Stage 1
checkpoint of SRResNet_2D and PoreSR respectively, passed with
--stage1_checkpoint, and then fine-tuned adversarially (Section 4.4).

Both stages select their output by best validation MS-SSIM (Section 4.5).
Stage 1 writes checkpoint_best.pth; Stage 2 writes checkpoint_gan_best.pth,
alongside a final-step snapshot.

Supports all six trained models. The four factorial cells of Section 4.1
vary input slice count and CBAM attention independently:

    Model name          Input slices   CBAM    Stage 2
    ----------------------------------------------------
    SRResNet_2D         1              no      no
    SRResNet_2D_CBAM    1              yes     no
    SRResNet_2_5D       5              no      no
    PoreSR              5              yes     no
    SRGAN_2D            1              no      yes
    PoreSR_GAN          5              yes     yes

Architecture configuration comes from GENERATOR_CONFIGS in models.generator,
so slice count and CBAM are never coupled in this script. Bicubic is a
non-learned baseline and is handled in evaluate.py only.

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
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MicroCTDataset
from losses.combined_loss import CombinedLoss
from models.discriminator import PatchDiscriminator
from models.generator import GENERATOR_CONFIGS, build_generator, count_parameters
from utils.checkpoint import CheckpointManager

# Adversarial variants and the Stage 1 backbone each is fine-tuned from.
STAGE1_BACKBONE = {
    "SRGAN_2D": "SRResNet_2D",
    "PoreSR_GAN": "PoreSR",
}
GAN_MODELS = tuple(STAGE1_BACKBONE)


def set_all_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed each DataLoader worker so augmentation order is reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_indices(path):
    """Load slice indices from a text file, ignoring comment lines."""
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if not line.startswith("#")]


def compute_metrics(sr, hr):
    """Compute PSNR, SSIM, and MS-SSIM for a single batch element."""
    from pytorch_msssim import ms_ssim
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    sr_np = sr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()

    psnr_val = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    ssim_val = structural_similarity(hr_np, sr_np, data_range=1.0)

    with torch.no_grad():
        ms_ssim_val = ms_ssim(
            sr.float(), hr.float(), data_range=1.0, size_average=True
        ).item()

    return {"psnr": psnr_val, "ssim": ssim_val, "ms_ssim": ms_ssim_val}


def validate(model, val_loader, device, mixed_precision=True):
    """Run validation and return averaged metrics."""
    model.eval()
    total_psnr, total_ssim, total_ms_ssim = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            lr_imgs = batch["lr"].to(device)
            hr_imgs = batch["hr"].to(device)

            with autocast(enabled=mixed_precision):
                sr_imgs = model(lr_imgs)

            for i in range(sr_imgs.shape[0]):
                metrics = compute_metrics(sr_imgs[i : i + 1], hr_imgs[i : i + 1])
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]
                total_ms_ssim += metrics["ms_ssim"]
                count += 1

    return {
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "ms_ssim": total_ms_ssim / count,
    }


def train_stage1(model, train_loader, val_loader, config, save_dir, model_name,
                 device):
    """
    Stage 1: Train generator with composite reconstruction loss.

    Uses Adam with linear warmup and cosine annealing under mixed precision.
    No gradient clipping is applied. The returned model is the best checkpoint
    by validation MS-SSIM.
    """
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
    )

    total_steps = config["total_steps"]
    warmup_steps = config["lr_warmup_steps"]
    lr_min = config["lr_min"]
    base_lr = config["learning_rate"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(lr_min / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedLoss(
        weight_l1=config["loss_l1"],
        weight_ms_ssim=config["loss_ms_ssim"],
        weight_gradient=config["loss_gradient"],
    )

    scaler = GradScaler(enabled=config["mixed_precision"])

    ckpt_mgr = CheckpointManager(
        save_dir=save_dir,
        interval_minutes=config["save_checkpoint_minutes"],
        keep_n=config["keep_n_checkpoints"],
    )

    start_step, start_epoch = ckpt_mgr.load(model, optimizer, scheduler)

    model.train()
    step = start_step
    epoch = start_epoch
    train_iter = iter(train_loader)

    pbar = tqdm(total=total_steps, initial=start_step,
                desc=model_name, unit="step")

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr_imgs = batch["lr"].to(device)
        hr_imgs = batch["hr"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config["mixed_precision"]):
            sr_imgs = model(lr_imgs)
            loss, _ = criterion(sr_imgs, hr_imgs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
        pbar.update(1)
        step += 1

        if step % config["val_every"] == 0 or step == total_steps:
            val_metrics = validate(model, val_loader, device,
                                   config["mixed_precision"])
            ckpt_mgr.log(
                f"Step {step} - Loss: {loss.item():.4f}, "
                f"PSNR: {val_metrics['psnr']:.2f}, "
                f"MS-SSIM: {val_metrics['ms_ssim']:.4f}"
            )
            ckpt_mgr.save(model, optimizer, scheduler, step, epoch,
                          loss.item(), val_metrics,
                          force=(step == total_steps))
            model.train()

    pbar.close()
    ckpt_mgr.log(f"Stage 1 complete: {step} steps")

    # Load the best Stage 1 checkpoint by validation MS-SSIM
    best_path = os.path.join(save_dir, "checkpoint_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_mgr.log(
            f"Loaded best model: MS-SSIM = {checkpoint['best_val_metric']:.4f}"
        )

    return model


def train_stage2(generator, train_loader, val_loader, config, save_dir,
                 model_name, device):
    """
    Stage 2: Adversarial fine-tuning with a PatchGAN discriminator.

    Uses hinge loss for the discriminator and a conservative adversarial
    weight (lambda_adv = 0.001) to limit structural disruption. The generator
    must already hold the selected Stage 1 weights of its backbone.

    Validation runs every config["val_every"] steps and the generator is saved
    to checkpoint_gan_best.pth whenever validation MS-SSIM improves, matching
    the selection criterion of Section 4.5. A final-step snapshot is also
    written. The returned generator is the best checkpoint, not the final one.
    """
    generator = generator.to(device)
    discriminator = PatchDiscriminator(in_channels=1).to(device)

    opt_g = optim.Adam(generator.parameters(),
                       lr=config["gan_lr_generator"], betas=(0.9, 0.999))
    opt_d = optim.Adam(discriminator.parameters(),
                       lr=config["gan_lr_discriminator"], betas=(0.9, 0.999))

    criterion_content = CombinedLoss(
        weight_l1=config["loss_l1"],
        weight_ms_ssim=config["loss_ms_ssim"],
        weight_gradient=config["loss_gradient"],
    )

    scaler_g = GradScaler(enabled=config["mixed_precision"])
    scaler_d = GradScaler(enabled=config["mixed_precision"])

    log_file = os.path.join(save_dir, "training_log.txt")

    def log(message):
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] {message}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log("Starting Stage 2 (GAN) training")

    generator.train()
    discriminator.train()
    train_iter = iter(train_loader)

    val_metrics = None
    best_val_metric = -float("inf")
    best_path = os.path.join(save_dir, "checkpoint_gan_best.pth")
    val_every = config["val_every"]

    pbar = tqdm(total=config["gan_steps"], desc=f"{model_name} GAN", unit="step")

    for step in range(config["gan_steps"]):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr_imgs = batch["lr"].to(device)
        hr_imgs = batch["hr"].to(device)

        # Train discriminator
        opt_d.zero_grad()
        with autocast(enabled=config["mixed_precision"]):
            with torch.no_grad():
                sr_imgs = generator(lr_imgs)
            pred_real = discriminator(hr_imgs)
            pred_fake = discriminator(sr_imgs.detach())
            loss_d = (torch.mean(F.relu(1.0 - pred_real))
                      + torch.mean(F.relu(1.0 + pred_fake)))

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # Train generator
        opt_g.zero_grad()
        with autocast(enabled=config["mixed_precision"]):
            sr_imgs = generator(lr_imgs)
            loss_content, _ = criterion_content(sr_imgs, hr_imgs)
            pred_fake = discriminator(sr_imgs)
            loss_adv = -torch.mean(pred_fake)
            loss_g = loss_content + config["gan_adversarial_weight"] * loss_adv

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        pbar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}")
        pbar.update(1)

        if (step + 1) % val_every == 0 or step == config["gan_steps"] - 1:
            val_metrics = validate(generator, val_loader, device,
                                   config["mixed_precision"])
            log(
                f"GAN Step {step + 1} - G: {loss_g.item():.4f}, "
                f"D: {loss_d.item():.4f}, "
                f"PSNR: {val_metrics['psnr']:.2f}, "
                f"MS-SSIM: {val_metrics['ms_ssim']:.4f}"
            )

            # Select on validation MS-SSIM, as in Stage 1 (Section 4.5).
            if val_metrics["ms_ssim"] > best_val_metric:
                best_val_metric = val_metrics["ms_ssim"]
                torch.save({
                    "step": step + 1,
                    "model_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_metric": best_val_metric,
                }, best_path)
                log(f"New best GAN checkpoint at step {step + 1}: "
                    f"MS-SSIM = {best_val_metric:.4f}")

            generator.train()

    pbar.close()

    # Final-step snapshot, retained for reference.
    gan_path = os.path.join(
        save_dir, f"generator_gan_step_{config['gan_steps']}.pth"
    )
    torch.save({
        "step": config["gan_steps"],
        "model_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "val_metrics": val_metrics,
    }, gan_path)
    log(f"Final-step snapshot saved: {gan_path}")

    # Return the selected checkpoint rather than the final step.
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device,
                                weights_only=False)
        generator.load_state_dict(checkpoint["model_state_dict"])
        log(f"Loaded best GAN checkpoint from step {checkpoint['step']}: "
            f"MS-SSIM = {checkpoint['best_val_metric']:.4f}")

    log("Stage 2 (GAN) training complete")

    return generator


def main():
    parser = argparse.ArgumentParser(
        description="PoreSR training. Trains one of the six learned models; "
                    "Bicubic is a non-learned baseline handled by evaluate.py."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration JSON file")
    parser.add_argument("--model", type=str, required=True,
                        choices=sorted(GENERATOR_CONFIGS),
                        help="Model to train")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--data_splits_dir", type=str, required=True,
                        help="Directory containing train/val/test indices")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint to initialise adversarial "
                             "fine-tuning from. Required for SRGAN_2D and "
                             "PoreSR_GAN; ignored otherwise.")
    args = parser.parse_args()

    if args.model in GAN_MODELS and args.stage1_checkpoint is None:
        parser.error(
            f"--stage1_checkpoint is required for {args.model}. It should be "
            f"the checkpoint_best.pth produced by training "
            f"{STAGE1_BACKBONE[args.model]}. Stage 2 fine-tunes that model for "
            f"5000 adversarial steps and does not retrain a backbone "
            f"(Section 4.4)."
        )
    if args.model not in GAN_MODELS and args.stage1_checkpoint is not None:
        parser.error(
            f"--stage1_checkpoint applies only to {' and '.join(GAN_MODELS)}."
        )

    if not torch.cuda.is_available():
        sys.exit(
            "Error: training requires a CUDA-capable GPU with at least 16 GB "
            "of memory. To verify the installation without a GPU, run "
            "examples/synthetic_demo.py instead."
        )
    device = torch.device("cuda")

    with open(args.config, "r") as f:
        config = json.load(f)

    set_all_seeds(config["seed"])

    os.makedirs(args.output_dir, exist_ok=True)

    # Architecture configuration comes from the registry, so input slice count
    # and CBAM are set independently for every model.
    arch = GENERATOR_CONFIGS[args.model]
    in_channels = arch["in_channels"]
    use_cbam = arch["use_cbam"]

    # The dataset supplies K adjacent slices, where K equals the generator's
    # input channel count.
    k = in_channels

    # Load data splits
    train_indices = load_indices(
        os.path.join(args.data_splits_dir, "train_indices.txt")
    )
    val_indices = load_indices(
        os.path.join(args.data_splits_dir, "val_indices.txt")
    )

    # Create datasets
    train_dataset = MicroCTDataset(
        slice_indices=train_indices,
        data_root_hr=config["data_root_hr"],
        data_root_lr=config["data_root_lr"],
        k=k,
        patch_size_hr=config["patch_size_hr"],
        patches_per_image=config["patches_per_image"],
        phase="train",
    )
    val_dataset = MicroCTDataset(
        slice_indices=val_indices,
        data_root_hr=config["data_root_hr"],
        data_root_lr=config["data_root_lr"],
        k=k,
        patch_size_hr=config["patch_size_hr"],
        patches_per_image=1,
        phase="val",
    )

    generator_seed = torch.Generator()
    generator_seed.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True,
        worker_init_fn=seed_worker, generator=generator_seed,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # Build model from the registry, allowing config overrides for the
    # backbone hyperparameters shared by all cells.
    model = build_generator(
        args.model,
        num_channels=config["num_channels"],
        num_blocks=config["num_residual_blocks"],
        upscale_factor=config["upscale_factor"],
    )

    print(f"Training {args.model} | in_channels={in_channels} | "
          f"CBAM={use_cbam} | K={k}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    if args.model in GAN_MODELS:
        # Stage 2 only. Load the selected Stage 1 weights of the backbone;
        # no Stage 1 training is performed here.
        backbone = STAGE1_BACKBONE[args.model]
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialised from {backbone} Stage 1 checkpoint: "
              f"{args.stage1_checkpoint}")
        if "best_val_metric" in checkpoint:
            print(f"  Stage 1 validation MS-SSIM: "
                  f"{checkpoint['best_val_metric']:.4f}")

        model = train_stage2(
            model, train_loader, val_loader, config, args.output_dir,
            args.model, device
        )
        print(
            f"Training complete. Evaluate this model using "
            f"checkpoint_gan_best.pth in {args.output_dir}"
        )
    else:
        model = train_stage1(
            model, train_loader, val_loader, config, args.output_dir,
            args.model, device
        )
        print(
            f"Training complete. Evaluate this model using "
            f"checkpoint_best.pth in {args.output_dir}"
        )


if __name__ == "__main__":
    main()
