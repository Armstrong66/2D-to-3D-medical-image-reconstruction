"""
Positional encoding for implicit neural representations.

Implements Fourier and SIREN-style positional encodings to combat spectral bias
in coordinate-MLPs.
"""
from __future__ import annotations
import torch
import torch.nn as nn


FOURIER = "fourier"
SIREN = "siren"


class FourierEncoding(nn.Module):
    """
    Fourier feature encoding for positional encoding.

    Encodes coordinates using sinusoidal features at multiple frequencies:
        gamma(p) = [p, sin(2^0*pi*p), cos(2^0*pi*p), ..., sin(2^{L-1}*pi*p), cos(2^{L-1}*pi*p)]

    Reference: NeRF, Mildenhall et al. ECCV 2020
    """

    def __init__(self, num_freqs: int = 6, log_scale: bool = True):
        """
        Args:
            num_freqs: Number of frequency bands
            log_scale: Whether to use log-spaced frequencies (2^0, 2^1, ..., 2^{L-1})
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.log_scale = log_scale

        if log_scale:
            self.freqs = torch.pow(2, torch.arange(num_freqs, dtype=torch.float32))
        else:
            self.freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input coordinates

        Returns:
            encoded: (B, N, D * (1 + 2*num_freqs)) encoded features
        """
        # Expand frequency dimension
        x_expanded = x.unsqueeze(-1)  # (B, N, D, 1)
        freqs = self.freqs.to(x.device)  # (num_freqs,)

        # Multiply: (B, N, D, num_freqs)
        scaled = x_expanded * freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Apply sin and cos
        sin_x = torch.sin(scaled)  # (B, N, D, num_freqs)
        cos_x = torch.cos(scaled)  # (B, N, D, num_freqs)

        # Flatten: (B, N, D * num_freqs)
        sin_flat = sin_x.flatten(-2, -1)
        cos_flat = cos_x.flatten(-2, -1)

        # Concatenate: (B, N, D * (1 + num_freqs))
        encoded = torch.cat([x, sin_flat, cos_flat], dim=-1)

        return encoded


class SIRENEncoding(nn.Module):
    """
    SIREN-style positional encoding.

    Similar to Fourier but uses omega_0 scaling for the first layer.
    """

    def __init__(self, num_freqs: int = 6, log_scale: bool = True, omega_0: float = 30.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.omega_0 = omega_0
        self.log_scale = log_scale

        if log_scale:
            self.freqs = torch.pow(2, torch.arange(num_freqs, dtype=torch.float32))
        else:
            self.freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SIREN-style encoding."""
        x_expanded = x.unsqueeze(-1)
        freqs = self.freqs.to(x.device)

        scaled = x_expanded * freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Apply sin with omega_0 scaling
        encoded = torch.sin(self.omega_0 * scaled)

        # Flatten
        encoded = encoded.flatten(-2, -1)

        return encoded


def build_positional_encoder(
    type: str,
    num_freqs: int = 6,
    log_scale: bool = True,
) -> nn.Module:
    """
    Factory to build positional encoder.

    Args:
        type: "fourier" or "siren"
        num_freqs: Number of frequency bands
        log_scale: Whether to use log-spaced frequencies

    Returns:
        Positional encoding module
    """
    if type == "fourier":
        return FourierEncoding(num_freqs=num_freqs, log_scale=log_scale)
    elif type == "siren":
        return SIRENEncoding(num_freqs=num_freqs, log_scale=log_scale)
    elif type is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown positional encoding type: {type}")
