import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.conv(x).view(B*T, -1)
        x = self.fc(x).view(B, T, -1)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, input_length=16000, output_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_length, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x.unsqueeze(1)