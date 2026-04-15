"""
PoreSR: PatchGAN Discriminator

PatchGAN discriminator with spectral normalisation for adversarial
fine-tuning of the PoreSR generator.

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

import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (Isola et al., 2018) with spectral normalisation
    (Miyato et al., 2018) on all convolutional weights.

    Produces a 15x15 spatial map of real/fake predictions for 256x256 input,
    each entry corresponding to a 70x70 pixel receptive field.

    Parameters
    ----------
    in_channels : int
        Number of input image channels. Default: 1 (greyscale micro-CT).
    """

    def __init__(self, in_channels=1):
        super().__init__()

        def disc_block(in_feat, out_feat):
            layers = [
                nn.utils.spectral_norm(nn.Conv2d(in_feat, out_feat, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, 64),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)
