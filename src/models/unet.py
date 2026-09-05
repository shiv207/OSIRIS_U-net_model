from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, depth=4):
        super().__init__()
        chs = [base_channels * (2**i) for i in range(depth)]
        self.inc = DoubleConv(in_channels, chs[0])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(depth - 1)])
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1]) for i in range(depth - 1)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)]
        )
        self.upconvs = nn.ModuleList([DoubleConv(chs[i], chs[i - 1]) for i in range(depth - 1, 0, -1)])
        self.outc = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for pool, down in zip(self.pools, self.downs):
            skips.append(down(pool(skips[-1])))
        x = skips[-1]
        for i, (up, conv) in enumerate(zip(self.ups, self.upconvs)):
            skip = skips[-(i + 2)]
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))
        return self.outc(x)
