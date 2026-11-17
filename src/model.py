"""Свёрточная нейросеть для предсказания расстояния до цели."""

import torch
import torch.nn as nn


class HeuristicCNN(nn.Module):
    def __init__(self, patch_size: int = 11):
        super().__init__()
        self.patch_size = patch_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        reduced = patch_size // 4

        # CNN-часть + два числа (dy, dx)
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 128),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W), offsets: (B, 2)."""
        x = self.features(x)
        x = self.cnn_head(x)
        x = torch.cat([x, offsets], dim=1)
        x = self.fc(x)
        return x.squeeze(-1)