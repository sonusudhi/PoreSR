"""
PoreSR: Checkpoint Manager

Handles automatic periodic checkpointing, best-model tracking, and
training resumption.

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

import os
import time
from datetime import datetime

import torch


class CheckpointManager:
    """
    Manages training checkpoints with periodic saving, best-model tracking,
    and automatic cleanup of old checkpoints.

    Parameters
    ----------
    save_dir : str
        Directory for checkpoint files.
    interval_minutes : int
        Minimum interval between periodic saves. Default: 30.
    keep_n : int
        Number of periodic checkpoints to retain. Default: 5.
    """

    def __init__(self, save_dir, interval_minutes=30, keep_n=5):
        self.save_dir = save_dir
        self.interval = interval_minutes * 60
        self.keep_n = keep_n
        self.last_save_time = time.time()
        self.best_val_metric = -float("inf")

        os.makedirs(save_dir, exist_ok=True)
        self.log_file = os.path.join(save_dir, "training_log.txt")

        with open(self.log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Training started: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 80}\n")

    def log(self, message):
        """Write a timestamped message to the log file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def save(self, model, optimizer, scheduler, step, epoch,
             train_loss, val_metrics, force=False):
        """
        Save checkpoint. Periodic saves occur at the configured interval.
        Best model is saved whenever validation metric improves.
        """
        current_time = time.time()
        should_save = force or (current_time - self.last_save_time >= self.interval)

        # Resolve the best metric before building the checkpoint, so that
        # checkpoint_best.pth records the metric that earned it and a resumed
        # run restores the correct threshold.
        current_metric = val_metrics.get("ms_ssim", -float("inf"))
        is_best = current_metric > self.best_val_metric
        if is_best:
            self.best_val_metric = current_metric

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "best_val_metric": self.best_val_metric,
            "timestamp": datetime.now().isoformat(),
        }

        # Always save latest
        torch.save(checkpoint, os.path.join(self.save_dir, "checkpoint_latest.pth"))

        # Periodic save
        if should_save:
            path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pth")
            torch.save(checkpoint, path)
            self.log(f"Saved checkpoint at step {step}")
            self.last_save_time = current_time
            self._cleanup()

        # Best model
        if is_best:
            torch.save(
                checkpoint, os.path.join(self.save_dir, "checkpoint_best.pth")
            )
            self.log(f"New best at step {step}: MS-SSIM = {current_metric:.4f}")

    def load(self, model, optimizer=None, scheduler=None):
        """
        Load the latest checkpoint if available.

        Returns
        -------
        start_step : int
            Step to resume from (0 if no checkpoint found).
        start_epoch : int
            Epoch to resume from.
        """
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")

        if not os.path.exists(latest_path):
            self.log("No checkpoint found, starting fresh")
            return 0, 0

        self.log(f"Loading checkpoint from {latest_path}")
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            latest_path, map_location=map_location, weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_metric = checkpoint["best_val_metric"]
        self.log(f"Resumed from step {checkpoint['step']}")

        return checkpoint["step"] + 1, checkpoint["epoch"]

    def _cleanup(self):
        """Remove old periodic checkpoints, keeping the most recent N."""
        checkpoints = sorted(
            [f for f in os.listdir(self.save_dir)
             if f.startswith("checkpoint_step_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for old in checkpoints[: -self.keep_n]:
            os.remove(os.path.join(self.save_dir, old))
