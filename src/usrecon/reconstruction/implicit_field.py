"""
Stage 3 - Implicit Neural Volume Reconstruction

Represents the reconstructed 3D volume as a continuous implicit function
f: R^3 -> R that maps 3D coordinates to intensity values.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from usrecon.reconstruction.positional_encoding import build_positional_encoder, FOURIER


class ImplicitFieldRegressor(nn.Module):
    """
    Coordinate-MLP for implicit volume reconstruction.

    Maps 3D coordinates to intensity values using a neural network.
    Optionally uses positional encoding to combat spectral bias.

    Architecture:
        input -> positional encoding -> MLP -> output

    Where MLP is:
        dim_in -> hidden -> hidden -> ... -> 1
    """

    def __init__(
        self,
        dim_in: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        positional_encoding: str = "fourier",
        pe_num_freqs: int = 6,
        pe_log_scale: bool = True,
    ):
        """
        Args:
            dim_in: Input dimension (3 for 3D points)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function ("relu" or "gelu")
            positional_encoding: Type of positional encoding ("fourier", "siren", or None)
            pe_num_freqs: Number of frequencies for Fourier encoding
            pe_log_scale: Whether to use log-spaced frequencies
        """
        super().__init__()
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.positional_encoding = positional_encoding

        # Build positional encoder
        if positional_encoding == "fourier":
            self.pe = build_positional_encoder("fourier", pe_num_freqs, pe_log_scale)
            pe_dim = dim_in * (1 + 2 * pe_num_freqs)  # [x, sin(2^0*x), cos(2^0*x), ...]
        elif positional_encoding == "siren":
            self.pe = build_positional_encoder("siren", pe_num_freqs, pe_log_scale)
            pe_dim = dim_in * (1 + 2 * pe_num_freqs)
        else:
            self.pe = None
            pe_dim = dim_in

        # Build MLP
        layers = []
        layers.append(nn.Linear(pe_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        layers.append(nn.Linear(hidden_dim, 1))  # Output intensity

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, name: str):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "sin":
            return SinLayer()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N_pts, 3) 3D coordinates

        Returns:
            intensities: (B, N_pts, 1) predicted intensities
        """
        # Apply positional encoding
        if self.pe is not None:
            x = self.pe(points)  # (B, N_pts, pe_dim)
        else:
            x = points  # (B, N_pts, 3)

        # Pass through MLP
        x = self.mlp(x)  # (B, N_pts, 1)

        return x


class SinLayer(nn.Module):
    """Sine activation for SIREN-style networks."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class ImplicitVolumeTrainer(nn.Module):
    """
    Trainer wrapper for implicit volume reconstruction.

    Handles forward pass, loss computation, and optional regularization.
    """

    def __init__(
        self,
        implicit_field: ImplicitFieldRegressor,
        point_cloud_compounder: nn.Module,
        lr: float = 0.0001,
    ):
        super().__init__()
        self.implicit_field = implicit_field
        self.point_cloud_compounder = point_cloud_compounder
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, frames: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct volume from frames and poses.

        Args:
            frames: (B, N, H, W) input frames
            poses: (B, N, 7) frame poses

        Returns:
            loss: Reconstruction loss
        """
        # Compound to point cloud
        points, intensities = self.point_cloud_compounder(frames, poses)

        # Predict intensities at point locations
        pred_intensities = self.implicit_field(points)  # (B, N_pts, 1)

        # Reconstruction loss (MSE)
        loss = torch.nn.functional.mse_loss(pred_intensities, intensities.unsqueeze(-1))

        return loss
