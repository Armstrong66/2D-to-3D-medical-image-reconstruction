"""
Common interface every Stage-0 encoder must satisfy, so Stage 1 (pose
regression) can consume either one without knowing which it's using.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch.nn as nn


class VisionEncoder(nn.Module, ABC):
    out_dim: int  # every subclass must set this in __init__

    @abstractmethod
    def forward(self, frames):
        """
        frames: (B, C, H, W) float tensor, C=1 for B-mode ultrasound.
        returns: (B, out_dim) pooled per-frame embedding.
        """
        raise NotImplementedError
