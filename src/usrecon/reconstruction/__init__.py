"""Reconstruction module (Stage 2-3)."""
# Placeholder for reconstruction submodules
# These will be implemented in later stages

from usrecon.reconstruction.compounding import compound_point_cloud
from usrecon.reconstruction.implicit_field import ImplicitFieldRegressor
from usrecon.reconstruction.positional_encoding import (
    build_positional_encoder,
    FOURIER,
    SIREN,
)

__all__ = [
    "compound_point_cloud",
    "ImplicitFieldRegressor",
    "build_positional_encoder",
    "FOURIER",
    "SIREN",
]
