"""Generator and discriminator architectures."""

from .generator import (
    CBAM,
    GENERATOR_CONFIGS,
    ResidualBlock,
    SRResNet,
    UpsampleBlock,
    build_generator,
    count_parameters,
)
from .discriminator import PatchDiscriminator

__all__ = [
    "CBAM",
    "GENERATOR_CONFIGS",
    "PatchDiscriminator",
    "ResidualBlock",
    "SRResNet",
    "UpsampleBlock",
    "build_generator",
    "count_parameters",
]
