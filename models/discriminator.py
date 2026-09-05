"""
PoreSR: PatchGAN Discriminator

PatchGAN discriminator with spectral normalisation for adversarial
fine-tuning of the PoreSR generator.

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

import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (Isola et al., 2017) with spectral normalisation
    (Miyato et al., 2018) applied to the four stride-2 convolutional blocks.
    The final 4x4 stride-1 convolution is not spectrally normalised.

    Produces a 15x15 spatial map of real/fake predictions for 256x256 input,
    each entry corresponding to a 94x94 pixel receptive field. The receptive
    field follows from four stride-2 4x4 convolutions and one stride-1 4x4
    convolution: 4 -> 10 -> 22 -> 46 -> 94.

    Used only for Stage 2 adversarial fine-tuning of SRGAN-2D and PoreSR-GAN
    (Section 4.4). Input is a single-channel HR or SR image, so in_channels
    stays 1 regardless of the generator's input slice count.

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
