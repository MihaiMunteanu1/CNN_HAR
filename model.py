"""
CNN models for Human Action Recognition on KTH dataset.
Input: HOG feature vectors extracted from 7-frame clips (same preprocessing as CSNN pipeline).
Output: 6 action classes — boxing, handclapping, handwaving, jogging, running, walking.

Three architectures are exposed via build_model:
  - "mlp"      : flat HOG → MLP                   (HARLinearNet)
  - "cnn"      : (T*C, H, W) → 2D CNN             (HARConvNet)
  - "temporal" : (T, C, H, W) → CNN encoder + BiGRU (HARTemporalNet)

Default HOG geometry per frame (cv2.HOGDescriptor on 64x128 window):
  C = 36, H = 15, W = 7  (T frames stacked along the temporal axis).
"""

import torch
import torch.nn as nn


KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
NUM_CLASSES = len(KTH_CLASSES)

HOG_C = 36
HOG_H = 15
HOG_W = 7


class HARLinearNet(nn.Module):
    """
    MLP baseline on flat HOG vector. Kept for backwards comparison with
    earlier runs.
    """

    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARConvNet(nn.Module):
    """
    2D CNN over the spatial HOG grid with frames stacked as channels.
    Input shape: (B, T*C, H, W) — e.g. (B, 7*36=252, 7, 3).
    """

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),               # (...) → (1,1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class HARTemporalNet(nn.Module):
    """
    Per-frame 2D CNN encoder + bidirectional GRU temporal head.
    Input shape: (B, T, C, H, W) — e.g. (B, 7, 36, 7, 3).
    """

    def __init__(self, c_per_frame: int = HOG_C, num_classes: int = NUM_CLASSES,
                 embed_dim: int = 192, gru_hidden: int = 128, dropout: float = 0.5):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(c_per_frame, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )

        # NB: dropout=0 pe GRU este intenționat — pe stack-ul ROCm/MIOpen al
        # GCP-ului, dropout-ul intern al GRU declanșează MIOpenDropoutHIP care
        # nu compilează (header rocrand_xorwow.h lipsă). Compensăm cu Dropout
        # extern și inter-layer prin temporal_dropout aplicat pe embeddings.
        self.temporal_dropout = nn.Dropout(0.3)
        self.temporal = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # Learned attention pooling over the T temporal steps.
        # Frames carrying the most movement get higher weight than static ones.
        self.attn_pool = nn.Sequential(
            nn.Linear(2 * gru_hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * gru_hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        emb = self.frame_encoder(x.reshape(B * T, C, H, W))   # (B*T, embed_dim)
        emb = emb.view(B, T, -1)
        emb = self.temporal_dropout(emb)
        out, _ = self.temporal(emb)                           # (B, T, 2*hidden)
        attn_logits = self.attn_pool(out)                     # (B, T, 1)
        attn = torch.softmax(attn_logits, dim=1)
        out = (out * attn).sum(dim=1)                         # (B, 2*hidden)
        return self.classifier(out)


class HARConv3DNet(nn.Module):
    """
    3D CNN that consumes the full (T, C, H, W) HOG tensor and learns
    spatio-temporal kernels jointly. Conceptually similar to HOG3D
    (Klaser et al., 2008), which is the strongest HOG-based baseline
    on KTH (~91%). The network learns its own temporal derivatives
    instead of relying on a frozen frame-encoder + recurrent head.

    Input shape: (B, T, C, H, W) — permuted internally to (B, C, T, H, W).
    """

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


def build_model(hog_shape: tuple, model_type: str = "cnn", dropout: float = 0.5) -> nn.Module:
    """
    Factory function.

    Args:
        hog_shape: shape of a single sample as returned by HOGDataset.
            - MLP : (flat_size,)
            - CNN : (T, C, H, W) — channels-as-time stacking is done inside
                    the model wrapper, since dataset returns (T, C, H, W).
            - Temporal : (T, C, H, W)
        model_type: 'mlp' | 'cnn' | 'temporal'
        dropout: classifier-head dropout rate.
    """
    mt = model_type.lower()

    if mt == "mlp":
        flat = 1
        for d in hog_shape:
            flat *= d
        return HARLinearNet(flat, dropout=dropout)

    if mt in ("cnn", "temporal", "conv3d"):
        if len(hog_shape) != 4:
            raise ValueError(
                f"{model_type} expects hog_shape (T, C, H, W); got {hog_shape}. "
                "Set as_image=True on HOGDataset."
            )
        T, C, _, _ = hog_shape
        if mt == "cnn":
            return _ChannelsAsTime(HARConvNet(in_channels=T * C, dropout=dropout))
        if mt == "conv3d":
            return HARConv3DNet(c_per_frame=C, dropout=dropout)
        return HARTemporalNet(c_per_frame=C, dropout=dropout)

    raise ValueError(f"Unknown model_type: {model_type!r}")


class _ChannelsAsTime(nn.Module):
    """
    Adapter: dataset yields (B, T, C, H, W); HARConvNet expects (B, T*C, H, W).
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
        return self.inner(x)