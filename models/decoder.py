import torch
import torch.nn as nn

class VideoDecoder(nn.Module):
    def __init__(self, input_dim=128, output_channels=3):
        super().__init__()
        # 编码器输出特征图大小为 8x8
        self.fc = nn.Linear(input_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, D = x.shape
        x = self.fc(x).view(B*T, 256, 8, 8)
        x = self.deconv(x).view(B, T, 3, 64, 64)
        return x

class AudioDecoder(nn.Module):
    def __init__(self, input_dim=128, output_length=16000):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_length)

    def forward(self, x):
        return self.fc(x)