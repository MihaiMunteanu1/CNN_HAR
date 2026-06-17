"""
3D CNN model for Human Action Recognition on KTH dataset.
Input: HOG feature vectors extracted from T-frame clips (same preprocessing as CSNN pipeline).
Output: 6 action classes — boxing, handclapping, handwaving, jogging, running, walking.

(T, C, H, W) → 3D CNN over the spatio-temporal HOG tensor (HARConv3DNet)
"""

import torch
import torch.nn as nn


KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
NUM_CLASSES = len(KTH_CLASSES)

HOG_C = 36
HOG_H = 15
HOG_W = 7


class HARConv3DNet(nn.Module):
    """3D CNN over the (T, C, H, W) HOG tensor. Three Conv3d blocks learn joint
    spatio-temporal kernels, an AdaptiveAvgPool3d collapses to a global 256-d
    descriptor, and a small classifier head maps it to the 6 KTH classes.
    The input (B, T, C, H, W) is permuted to (B, C, T, H, W) for Conv3d."""

    def __init__(self, c_per_frame: int = HOG_C, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: spatio-temporal conv on raw HOG cells
            nn.Conv3d(c_per_frame, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout3d(0.25),

            # Block 2: deeper temporal kernel
            nn.Conv3d(128, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout3d(0.25),

            # Block 3: collapse to global descriptor
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.classifier(self.features(x))


def build_model(hog_shape: tuple, dropout: float = 0.4) -> nn.Module:
    if len(hog_shape) != 4:
        raise ValueError(
            f"conv3d expects hog_shape (T, C, H, W); got {hog_shape}. "
            "Set as_image=True on HOGDataset."
        )
    _, C, _, _ = hog_shape
    return HARConv3DNet(c_per_frame=C, dropout=dropout)