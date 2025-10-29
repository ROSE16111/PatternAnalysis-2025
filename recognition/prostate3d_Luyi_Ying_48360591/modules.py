# -*- coding: utf-8 -*-
"""
3D U-Net baseline for Prostate segmentation.
中文注释：这是最小可运行的3D U-Net，层数和通道数偏小，方便显存有限时先跑通。
后续可升级：残差块/注意力/深监督 = Improved UNet3D（Hard）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3d(nn.Module):
    """两次(Conv3d+IN+LeakyReLU)，InstanceNorm对3D医学图像较稳定"""
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
    """下采样（MaxPool3d）+ 卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ConvBlock3d(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Up3d(nn.Module):
    """反卷积上采样 + 与skip连接后再卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 上采样将通道数减半，方便与skip concat
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock3d(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 若尺寸因奇偶差异不一致，则pad到与skip一致
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
    in_ch: 输入通道（CT/MRI灰度一般=1）
    num_classes: 类别数（含背景）；训练用 CrossEntropy
    base: 基础通道数（小显存可设16/32）
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
