"""Reconstruction module (Stage 2-3)."""
# Placeholder for reconstruction submodules
# These will be implemented in later stages

from .reconstruction.compounding import compound_point_cloud
from .reconstruction.implicit_field import ImplicitFieldRegressor
from .reconstruction.positional_encoding import (
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
