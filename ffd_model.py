#!/usr/bin/env python3
"""
ffd_model.py

- Small 3D U-Net that predicts ΔG for an (nx,ny,nz) FFD lattice.
- Head outputs 3*nx*ny*nz channels; we reshape to (nx,ny,nz,3).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_c, out_c):
    return nn.Conv3d(in_c, out_c, 3, padding=1)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(in_c, out_c), nn.InstanceNorm3d(out_c), nn.LeakyReLU(0.1, inplace=True),
            conv3x3(out_c, out_c), nn.InstanceNorm3d(out_c), nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, 2, stride=2)
        self.conv = DownBlock(in_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed for odd dims
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2, diffZ//2, diffZ - diffZ//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet3D_FFD(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 16,
                 lattice_size: Tuple[int, int, int] = (6,6,6)):
        super().__init__()
        self.nx,self.ny,self.nz = lattice_size
        out_channels = 3 * self.nx * self.ny * self.nz

        self.inc = DownBlock(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DownBlock(base_channels, base_channels*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DownBlock(base_channels*2, base_channels*4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DownBlock(base_channels*4, base_channels*8))

        self.bottom = DownBlock(base_channels*8, base_channels*16)

        self.up3 = UpBlock(base_channels*16, base_channels*8)
        self.up2 = UpBlock(base_channels*8, base_channels*4)
        self.up1 = UpBlock(base_channels*4, base_channels*2)
        self.up0 = UpBlock(base_channels*2, base_channels)

        self.head = nn.Sequential(
            conv3x3(base_channels, base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(base_channels, out_channels, kernel_size=1)
        )

        # optional: scale prediction to mm range ~[-3,3]
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xb = self.bottom(x3)

        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        y = self.head(y0)  # (B, 3*nx*ny*nz, D,H,W)

        # Global average to a single ΔG set per volume (weakly conditioned on whole image)
        y = y.mean(dim=[2,3,4])  # (B, 3*G)
        B = y.shape[0]
        y = y.view(B, self.nx, self.ny, self.nz, 3)
        return self.scale * y
