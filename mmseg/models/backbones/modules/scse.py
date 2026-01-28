import torch.nn as nn


class cSE(nn.Module):
    """Channel Squeeze & Excitation (cSE)"""
    def __init__(self, channels: int, reduction: int = 16, min_channels: int = 4):
        super().__init__()
        mid = max(channels // reduction, min_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))  # [B,C,1,1]
        return x * w


class sSE(nn.Module):
    """Spatial Squeeze & Excitation (sSE)"""
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.spatial(x)  # [B,1,H,W]
        return x * w


class scSE(nn.Module):
    """Concurrent Spatial and Channel SE (scSE): scSE(x) = cSE(x) + sSE(x)"""
    def __init__(self, channels: int, reduction: int = 16, min_channels: int = 4):
        super().__init__()
        self.cse = cSE(channels, reduction=reduction, min_channels=min_channels)
        self.sse = sSE(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)
