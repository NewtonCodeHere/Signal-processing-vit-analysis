"""
Vertically Patched Vision Transformer (ViT) Architecture (Model 4).
Faithfully recreates the Project I & II architecture documented in TCE presentations:
Input Spectrogram (128x128)
-> Convolutional Stem (16 -> 32 -> 64 -> 32 -> 16 filters, 3x3 kernels)
-> Vertical Patching (128x1 time-slice patches)
-> Linear Token Projection (256 dimensions)
-> 1D Positional Encoding
-> Transformer Encoder (Multi-Head Self-Attention)
-> MLP Emotion Classifier Head
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any


class ConvStem(nn.Module):
    """
    Convolutional stem extracting local time-frequency features.
    Filters: 16 -> 32 -> 64 -> 32 -> 16 using 3x3 kernels.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        return x  # Shape: (B, 16, 128, 128)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequential speech time order.
    """
    def __init__(self, d_model: int = 256, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class VerticallyPatchedViT(nn.Module):
    """
    Vertically Patched Vision Transformer for Speech Emotion Recognition.
    """
    def __init__(
        self,
        num_classes: int = 8,
        in_mels: int = 128,
        in_frames: int = 128,
        conv_out_channels: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv_stem = ConvStem(in_channels=1)

        # Vertical patch size: (in_mels, 1) across conv_out_channels
        # Flattened patch vector dimension = in_mels * conv_out_channels = 128 * 16 = 2048
        self.patch_dim = in_mels * conv_out_channels
        self.token_projection = nn.Linear(self.patch_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=in_frames)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Emotion Classifier Head
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-Mel spectrogram tensor of shape (B, 1, 128, 128)
        Returns:
            Logits of shape (B, num_classes)
        """
        # 1. Convolutional stem for local time-frequency features
        feat = self.conv_stem(x)  # (B, C=16, F=128, T=128)

        # 2. Vertical patchify: slice into vertical time strips (each patch = 1 time frame)
        B, C, F, T = feat.shape
        # Permute to (B, T, C, F) and flatten channels & frequencies into patch vectors
        patches = feat.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, 128, 2048)

        # 3. Token embedding
        tokens = self.token_projection(patches)  # (B, 128, d_model=256)

        # 4. Positional encoding
        tokens = self.pos_encoder(tokens)

        # 5. Transformer encoder self-attention
        encoded = self.transformer_encoder(tokens)  # (B, 128, 256)

        # Global temporal average pooling across time patches
        pooled = torch.mean(encoded, dim=1)  # (B, 256)

        # 6. MLP classifier
        logits = self.mlp_head(pooled)
        return logits
