"""
PoreSR: Generator Architecture

Modified SRResNet with optional CBAM attention for 2D and 2.5D
super-resolution of rock micro-CT images.

A single SRResNet class implements all four cells of the 2x2 factorial
described in Section 4.1 of the paper. The cells differ only in the number
of input slices (1 or 5) and the presence of CBAM attention:

    Model name          Input slices   CBAM    Trainable parameters
    -----------------------------------------------------------------
    SRResNet_2D         1              no       1,524,500
    SRResNet_2D_CBAM    1              yes      1,534,260
    SRResNet_2_5D       5              no       1,545,236
    PoreSR              5              yes      1,554,996

Trainable parameter count therefore varies by 2.0% across the four cells.
SRGAN_2D and PoreSR_GAN use the SRResNet_2D and PoreSR generators
respectively, fine-tuned adversarially in Stage 2 (see train.py).

Run this file directly to print the parameter counts for all four cells:

    python models/generator.py

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


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Applies sequential channel and spatial attention to refine feature maps.
    The 7x7 spatial attention kernel operates at LR scale and therefore
    corresponds to 28x28 pixels of HR-equivalent spatial support, about
    132 um, some 6.5 times the HR D50 of 20.18 um for this sample. The
    attention window therefore spans several characteristic pore-throat
    widths rather than a single throat.

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

    Batch normalisation is removed following Lim et al. (2017). Batch
    normalisation rescales feature magnitude statistics, which is expected to
    be particularly detrimental at micro-CT pore-grain interfaces where
    grey-level distributions carry segmentation-critical meaning.

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

    The four factorial cells of Section 4.1 are constructed as:

        SRResNet_2D        SRResNet(in_channels=1, use_cbam=False)
        SRResNet_2D_CBAM   SRResNet(in_channels=1, use_cbam=True)
        SRResNet_2_5D      SRResNet(in_channels=5, use_cbam=False)
        PoreSR             SRResNet(in_channels=5, use_cbam=True)

    Use build_generator() rather than constructing these directly, so that
    model names stay consistent with train.py and evaluate.py.

    Input shape is (B, in_channels, H, W); output is always (B, 1, 4H, 4W),
    since the 2.5D variants reconstruct only the centre slice of the input
    stack.

    Parameters
    ----------
    in_channels : int
        Number of input channels. 1 for 2D, 5 for 2.5D (K adjacent slices).
    num_channels : int
        Number of feature channels in residual blocks. Default: 64.
    num_blocks : int
        Number of residual blocks. Default: 16.
    upscale_factor : int
        Total spatial upscaling factor. Must be a power of two. Default: 4,
        achieved through log2(upscale_factor) sequential 2x PixelShuffle
        blocks.
    use_cbam : bool
        Whether to include CBAM attention in residual blocks. Default: False.
    """

    def __init__(self, in_channels=1, num_channels=64, num_blocks=16,
                 upscale_factor=4, use_cbam=False):
        super().__init__()

        if upscale_factor < 2 or (upscale_factor & (upscale_factor - 1)) != 0:
            raise ValueError(
                f"upscale_factor must be a power of two and at least 2, "
                f"got {upscale_factor}"
            )

        self.in_channels = in_channels
        self.use_cbam = use_cbam
        self.upscale_factor = upscale_factor

        self.conv_input = nn.Conv2d(in_channels, num_channels, 9, 1, 4)
        self.prelu_input = nn.PReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, use_cbam) for _ in range(num_blocks)]
        )

        self.conv_mid = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        # Sequential 2x upsampling blocks achieve the requested magnification.
        # The default upscale_factor of 4 gives two blocks, as used throughout
        # the paper.
        num_upsample_blocks = int(upscale_factor).bit_length() - 1
        self.upsample = nn.Sequential(
            *[UpsampleBlock(num_channels, 2) for _ in range(num_upsample_blocks)]
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


# Model registry. Each entry gives the generator configuration for one named
# method. SRGAN_2D and PoreSR_GAN share the generator of their Stage 1
# backbone and differ only in Stage 2 adversarial fine-tuning.
GENERATOR_CONFIGS = {
    # 2x2 factorial (Section 4.1)
    "SRResNet_2D":      {"in_channels": 1, "use_cbam": False},
    "SRResNet_2D_CBAM": {"in_channels": 1, "use_cbam": True},
    "SRResNet_2_5D":    {"in_channels": 5, "use_cbam": False},
    "PoreSR":           {"in_channels": 5, "use_cbam": True},
    # Adversarial variants (Section 4.4)
    "SRGAN_2D":         {"in_channels": 1, "use_cbam": False},
    "PoreSR_GAN":       {"in_channels": 5, "use_cbam": True},
}


def build_generator(model_name, **overrides):
    """
    Construct the generator for a named method.

    Parameters
    ----------
    model_name : str
        One of the keys of GENERATOR_CONFIGS.
    **overrides
        Optional keyword arguments passed to SRResNet, overriding the
        registry defaults (for example num_blocks or num_channels).

    Returns
    -------
    SRResNet
        The configured generator.

    Raises
    ------
    ValueError
        If model_name is not a recognised method.
    """
    if model_name not in GENERATOR_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Expected one of: "
            f"{', '.join(sorted(GENERATOR_CONFIGS))}"
        )
    config = dict(GENERATOR_CONFIGS[model_name])
    config.update(overrides)
    return SRResNet(**config)


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Print the trainable parameter counts of the four factorial cells.
    # These reproduce the values reported in Sections 4.1 and 4.3.
    factorial = [
        "SRResNet_2D",
        "SRResNet_2D_CBAM",
        "SRResNet_2_5D",
        "PoreSR",
    ]
    counts = {name: count_parameters(build_generator(name))
              for name in factorial}

    print(f"{'Model':<20}{'Input slices':>14}{'CBAM':>7}{'Parameters':>14}")
    print("-" * 55)
    for name in factorial:
        cfg = GENERATOR_CONFIGS[name]
        print(f"{name:<20}{cfg['in_channels']:>14}"
              f"{'yes' if cfg['use_cbam'] else 'no':>7}{counts[name]:>14,}")

    cbam_cost = counts["PoreSR"] - counts["SRResNet_2_5D"]
    slice_cost = counts["SRResNet_2_5D"] - counts["SRResNet_2D"]
    spread = (max(counts.values()) - min(counts.values())) / min(counts.values())

    print("-" * 55)
    print(f"CBAM contributes             {cbam_cost:,} parameters "
          f"({cbam_cost / counts['PoreSR'] * 100:.2f}% of PoreSR)")
    print(f"Five-slice input contributes {slice_cost:,} parameters")
    print(f"Spread across factorial cells {spread * 100:.1f}%")
