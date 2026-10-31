"""
CNN-BiLSTM Architecture (Model 6).
Recreates the Project I & II high-accuracy model (87% test accuracy on RAVDESS + CREMAD + TESS):
Convolutional 2D blocks -> Bidirectional LSTM layers -> Fully Connected Emotion Classifier.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class CNNBiLSTM(nn.Module):
    """
    Hybrid Convolutional Neural Network and Bidirectional LSTM for SER.
    """
    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, 64, 64)
            nn.Dropout2d(0.2)
        )

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, 32, 32)
            nn.Dropout2d(0.2)
        )

        # Conv Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Reduce frequency, preserve time: (B, 128, 16, 32)
            nn.Dropout2d(0.3)
        )

        # Bidirectional LSTM
        # Input features per time step = 128 channels * 16 frequency bins = 2048
        self.lstm_in_dim = 128 * 16
        self.bilstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
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
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # Shape: (B, 128, 16, 32)

        B, C, F, T = x.shape
        # Permute to (B, T, C * F)
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        # Bidirectional LSTM forward
        lstm_out, _ = self.bilstm(x_seq)  # (B, T, hidden_dim * 2)

        # Temporal attention / mean pooling
        pooled = torch.mean(lstm_out, dim=1)  # (B, hidden_dim * 2)

        # Emotion classification logits
        logits = self.classifier(pooled)
        return logits
