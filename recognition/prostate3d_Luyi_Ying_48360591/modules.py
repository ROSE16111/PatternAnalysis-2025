# -*- coding: utf-8 -*-
"""
3D U-Net baseline for Prostate segmentation.
This is the minimum viable 3D U-Net, with a relatively small number of layers and channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3d(nn.Module):
    """InstanceNorm is more stable for 3D medical images after two (Conv3d+IN+LeakyReLU) operations."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down3d(nn.Module):
    """Downsampling (MaxPool3d) + Convolutional Blocks"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ConvBlock3d(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Up3d(nn.Module):
    """Deconvolution upsampling + convolution block after skipping"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsampling halves the number of channels, making it easier to concatenate with skip.
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock3d(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # If the dimensions are inconsistent due to parity differences, then pad to and skip should be consistent.
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(x, [dx // 2, dx - dx // 2,
                      dy // 2, dy - dy // 2,
                      dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet3D(nn.Module):
    """
    Minimal 3D U-Net
    in_ch: Input channel (CT/MRI grayscale is generally = 1)
    num_classes: Number of categories (including background); training use CrossEntropy
    base: Base number of channels (can be set to 16/32 for small video memory)
    """
    def __init__(self, in_ch=1, num_classes=6, base=16):
        super().__init__()
        self.inc = ConvBlock3d(in_ch, base)      # 16
        self.d1 = Down3d(base, base * 2)         # 32
        self.d2 = Down3d(base * 2, base * 4)     # 64
        self.d3 = Down3d(base * 4, base * 8)     # 128
        self.bottleneck = ConvBlock3d(base * 8, base * 16)  # 256
        self.u3 = Up3d(base * 16, base * 8)      # 256->128
        self.u2 = Up3d(base * 8, base * 4)       # 128->64
        self.u1 = Up3d(base * 4, base * 2)       # 64->32
        self.u0 = Up3d(base * 2, base)           # 32->16
        self.outc = nn.Conv3d(base, num_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        xb = self.bottleneck(x3)
        x = self.u3(xb, x3)
        x = self.u2(x, x2)
        x = self.u1(x, x1)
        x = self.u0(x, x0)
        return self.outc(x)
