"""
PoreSR: Combined Reconstruction Loss

Composite loss function: L1 + MS-SSIM + Gradient (Sobel edge loss).
Mixed-precision compatible.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class CombinedLoss(nn.Module):
    """
    Composite reconstruction loss for micro-CT super-resolution.

    L_total = w_l1 * L1 + w_ms_ssim * (1 - MS-SSIM) + w_grad * L_grad

    This is Eq. (1) of the paper, with default weights lambda_1 = 1.0,
    lambda_2 = 0.1 and lambda_3 = 0.01. The same loss and weights are used for
    all four cells of the factorial and for the content term of Stage 2.

    The L1 term provides robust pixel-level fidelity under the heavy-tailed
    error distribution dominated by high-contrast pore-grain boundaries.
    MS-SSIM penalises structural dissimilarities at five Gaussian pyramid
    scales. The gradient loss penalises edge displacement at pore-grain
    interfaces through horizontal and vertical Sobel filters.

    Parameters
    ----------
    weight_l1 : float
        Weight for L1 loss. Default: 1.0.
    weight_ms_ssim : float
        Weight for MS-SSIM loss. Default: 0.1.
    weight_gradient : float
        Weight for gradient (Sobel edge) loss. Default: 0.01.
    """

    def __init__(self, weight_l1=1.0, weight_ms_ssim=0.1, weight_gradient=0.01):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_ms_ssim = weight_ms_ssim
        self.weight_gradient = weight_gradient
        self.l1_loss = nn.L1Loss()

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient_loss(self, pred, target):
        """
        L1 loss on horizontal and vertical Sobel gradients.

        The two directional terms are summed, not averaged, so the effective
        weight on each direction is weight_gradient.
        """
        sx = self.sobel_x.type_as(pred)
        sy = self.sobel_y.type_as(pred)

        pred_gx = F.conv2d(pred, sx, padding=1)
        pred_gy = F.conv2d(pred, sy, padding=1)
        target_gx = F.conv2d(target, sx, padding=1)
        target_gy = F.conv2d(target, sy, padding=1)

        return self.l1_loss(pred_gx, target_gx) + self.l1_loss(pred_gy, target_gy)

    def forward(self, pred, target):
        """
        Compute composite loss.

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of L1, MS-SSIM, and gradient losses.
        loss_dict : dict
            Individual loss components for logging.
        """
        loss_l1 = self.l1_loss(pred, target)

        # MS-SSIM is numerically unstable in reduced precision, so evaluate it
        # in float32. Tensor.float() is a no-op on tensors that are already
        # float32, so this covers float16 and bfloat16 autocast alike without
        # changing float32 behaviour.
        loss_ms_ssim = 1 - ms_ssim(
            pred.float(), target.float(), data_range=1.0, size_average=True
        )

        loss_grad = self.gradient_loss(pred, target)

        total_loss = (
            self.weight_l1 * loss_l1
            + self.weight_ms_ssim * loss_ms_ssim
            + self.weight_gradient * loss_grad
        )

        loss_dict = {
            "l1": loss_l1.item(),
            "ms_ssim": loss_ms_ssim.item(),
            "gradient": loss_grad.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
