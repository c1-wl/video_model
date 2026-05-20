import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128, image_size=(64,64)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 16->8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.conv(x).squeeze(-1).squeeze(-1)   # (B, 128)
        return self.fc(feat)   # (B, output_dim)