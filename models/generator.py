"""
PoreSR: Generator Architecture

Modified SRResNet with optional CBAM attention for 2D and 2.5D
super-resolution of rock micro-CT images.

Authors:
    Sonu Sudhikumar Seena (1,2), Lin Ma (1), Anirban Chakraborty (2), Jingyue Hao (1)

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

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Applies sequential channel and spatial attention to refine feature maps.
    The 7x7 spatial attention kernel at LR scale corresponds to a 28x28 pixel
    HR receptive field (132 um), ensuring attention operates at the scale of
    entire pore throats in Sherwood sandstone.

    Parameters
    ----------
    channels : int
        Number of input feature channels.
    reduction : int
        Channel reduction ratio for the bottleneck MLP. Default: 16.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # Spatial attention (7x7 kernel)
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_att

        return x


class ResidualBlock(nn.Module):
    """
    Residual block with optional CBAM attention.

    Batch normalisation is removed following Lim et al. (2017), as it
    distorts feature magnitude statistics at pore-grain interfaces where
    grey-level distributions carry segmentation-critical radiometric meaning.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    use_cbam : bool
        Whether to apply CBAM attention after the residual path.
    """

    def __init__(self, channels, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.cbam = CBAM(channels) if use_cbam else None

    def forward(self, x):
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        if self.cbam is not None:
            out = self.cbam(out)
        return out + residual


class UpsampleBlock(nn.Module):
    """
    Sub-pixel upsampling block using PixelShuffle (Shi et al., 2016).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    upscale_factor : int
        Spatial upsampling factor per block.
    """

    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * (upscale_factor ** 2), 3, 1, 1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SRResNet(nn.Module):
    """
    Super-Resolution Residual Network (Ledig et al., 2017) with domain-specific
    modifications for rock micro-CT super-resolution.

    Modifications from standard SRResNet:
        (i)   Input channel count configurable for 2.5D multi-slice stacks.
        (ii)  Optional CBAM attention in each residual block.
        (iii) Batch normalisation removed throughout.

    For 2D operation, set in_channels=1. For 2.5D (PoreSR), set in_channels=5.

    Parameters
    ----------
    in_channels : int
        Number of input channels. 1 for 2D, 5 for 2.5D (K adjacent slices).
    num_channels : int
        Number of feature channels in residual blocks. Default: 64.
    num_blocks : int
        Number of residual blocks. Default: 16.
    upscale_factor : int
        Total spatial upscaling factor. Default: 4 (achieved via 2x2).
    use_cbam : bool
        Whether to include CBAM attention in residual blocks. Default: False.
    """

    def __init__(self, in_channels=1, num_channels=64, num_blocks=16,
                 upscale_factor=4, use_cbam=False):
        super().__init__()

        self.conv_input = nn.Conv2d(in_channels, num_channels, 9, 1, 4)
        self.prelu_input = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, use_cbam) for _ in range(num_blocks)]
        )

        self.conv_mid = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        # Two sequential 2x upsampling blocks achieve 4x magnification
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            UpsampleBlock(num_channels, 2),
        )

        self.conv_output = nn.Conv2d(num_channels, 1, 9, 1, 4)

    def forward(self, x):
        out = self.prelu_input(self.conv_input(x))
        residual = out
        out = self.residual_blocks(out)
        out = self.conv_mid(out)
        out = out + residual
        out = self.upsample(out)
        out = self.conv_output(out)
        return out
