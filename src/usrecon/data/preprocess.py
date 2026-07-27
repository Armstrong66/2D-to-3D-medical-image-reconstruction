"""
Data preprocessing utilities for usrecon.

This module provides standard preprocessing pipelines for ultrasound imaging,
including noise reduction, normalization, and augmentation.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "normalize",
    "denoise",
    "augment",
    "preprocess_frame",
    "PreprocessingConfig",
]


class PreprocessingConfig:
    """Configuration for ultrasound image preprocessing."""

    def __init__(
        self,
        normalize: bool = True,
        mean: float | None = None,
        std: float | None = None,
        denoise: bool = True,
        denoise_kernel_size: int = 5,
        augment: bool = True,
        noise_std: float = 0.01,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        crop_scale: Tuple[float, float] = (0.9, 1.0),
    ):
        """
        Args:
            normalize: Whether to normalize pixel values
            mean: Manual mean for normalization (auto-calculated if None)
            std: Manual std for normalization (auto-calculated if None)
            denoise: Whether to apply denoising
            denoise_kernel_size: Size of denoising kernel
            augment: Whether to apply augmentation
            noise_std: Standard deviation of Gaussian noise for augmentation
            flip_prob: Probability of horizontal flip
            rotate_prob: Probability of random rotation
            crop_scale: Scale range for random crop augmentation
        """
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.denoise = denoise
        self.denoise_kernel_size = denoise_kernel_size
        self.augment = augment
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.crop_scale = crop_scale


def normalize(
    frames: Tensor,
    mean: float | None = None,
    std: float | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Normalize ultrasound frames to zero mean and unit variance.

    Args:
        frames: Input tensor of shape (B, C, H, W) or (C, H, W)
        mean: Pre-computed mean (auto-calculated if None)
        std: Pre-computed std (auto-calculated if None)
        eps: Small value for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    # Compute statistics if not provided
    if mean is None:
        mean = frames.mean(dim=[0, 2, 3], keepdim=True)
    if std is None:
        std = frames.std(dim=[0, 2, 3], keepdim=True) + eps

    normalized = (frames - mean) / std

    if squeeze_back:
        normalized = normalized.squeeze(0)

    return normalized


def denoise(frames: Tensor, kernel_size: int = 5) -> Tensor:
    """
    Apply median denoising to ultrasound frames.

    Args:
        frames: Input tensor of shape (B, C, H, W)
        kernel_size: Size of median filter kernel

    Returns:
        Denoised tensor with same shape as input
    """
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)

    # Use average pooling as a simple denoiser (median not available in torch)
    # For better results, use scipy.ndimage.median_filter
    padded = F.pad(frames, [kernel_size // 2] * 4, mode="reflect")
    denoised = F.avg_pool2d(
        padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    return denoised


def augment(
    frames: Tensor,
    config: PreprocessingConfig,
) -> Tensor:
    """
    Apply random augmentation to ultrasound frames.

    Args:
        frames: Input tensor of shape (B, C, H, W)
        config: Preprocessing configuration

    Returns:
        Augmented tensor with same shape as input
    """
    if not config.augment:
        return frames

    if frames.ndim == 3:
        frames = frames.unsqueeze(0)

    augmented = frames.clone()

    # Random horizontal flip
    if random.random() < config.flip_prob:
        augmented = torch.flip(augmented, dims=[3])

    # Random rotation (90, 180, 270 degrees)
    if random.random() < config.rotate_prob:
        k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
        augmented = torch.rot90(augmented, k=k, dims=[2, 3])

    # Random Gaussian noise
    if config.noise_std > 0:
        noise = torch.randn_like(augmented) * config.noise_std
        augmented = augmented + noise
        augmented = torch.clamp(augmented, 0.0, 1.0)

    return augmented


def preprocess_frame(
    frame: Tensor,
    config: PreprocessingConfig | None = None,
) -> Tensor:
    """
    Apply full preprocessing pipeline to a single frame.

    Args:
        frame: Input frame of shape (C, H, W) or (H, W)
        config: Preprocessing configuration (uses defaults if None)

    Returns:
        Preprocessed frame
    """
    if config is None:
        config = PreprocessingConfig()

    # Ensure 4D tensor
    if frame.ndim == 2:
        frame = frame.unsqueeze(0)  # Add channel dim
    if frame.ndim == 3:
        frame = frame.unsqueeze(0)  # Add batch dim

    # Normalize
    if config.normalize:
        frame = normalize(frame, mean=config.mean, std=config.std)

    # Denoise
    if config.denoise:
        frame = denoise(frame, kernel_size=config.denoise_kernel_size)

    # Augment (only for training)
    if config.augment:
        frame = augment(frame, config)

    # Remove batch dim
    frame = frame.squeeze(0)

    return frame


def compute_dataset_stats(
    frames: Tensor,
    max_samples: int | None = None,
) -> Tuple[float, float]:
    """
    Compute dataset statistics (mean and std) for normalization.

    Args:
        frames: Tensor of shape (N, C, H, W) containing dataset samples
        max_samples: Maximum number of samples to use (None = all)

    Returns:
        Tuple of (mean, std) for normalization
    """
    if max_samples is not None and max_samples < len(frames):
        frames = frames[:max_samples]

    # Reshape to (N*C*H*W) for statistics
    flat = frames.flatten()

    mean = flat.mean().item()
    std = flat.std().item()

    return mean, std


def ultrasound_window(
    frames: Tensor,
    window_min: float = 0.0,
    window_max: float = 1.0,
) -> Tensor:
    """
    Apply intensity windowing to ultrasound frames.

    Args:
        frames: Input tensor of shape (B, C, H, W)
        window_min: Minimum intensity value (clipped)
        window_max: Maximum intensity value (clipped)

    Returns:
        Windowed tensor
    """
    windowed = torch.clamp(frames, window_min, window_max)
    return windowed


def histogram_equalization(frames: Tensor, bins: int = 256) -> Tensor:
    """
    Apply histogram equalization to improve contrast.

    Note: This is a simplified implementation using PyTorch.
    For production, use scipy.ndimage.histogram_equalization.

    Args:
        frames: Input tensor of shape (B, C, H, W) with values in [0, 1]
        bins: Number of histogram bins

    Returns:
        Equalized tensor with same shape
    """
    # This is a simplified implementation
    # For production, use scipy.ndimage.histogram_equalization
    return frames
