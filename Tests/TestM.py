import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as Fu

class D_CBR(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(D_CBR, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unit(x)

class D_BRC(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(D_BRC, self).__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.unit(x)

class BRC(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(BRC, self).__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.unit(x)

class UpSampling(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(UpSampling, self).__init__()
        self.opt = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        return self.opt(x)

class BilinearInterpolation(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, mode="bilinear", align_corners=True):
        super(BilinearInterpolation, self).__init__()
        self.opt = nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.opt(x))

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = Fu.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class ECABlock(nn.Module):

    def __init__(self, in_channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.t = int(abs(math.log2(in_channels) + b) / gamma)
        self.ks = self.t if(self.t % 2) else self.t + 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, self.ks, padding=self.ks//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.gap(x).squeeze(-1).transpose(-1, -2)
        x2 = self.conv1d(x1).transpose(-1, -2).unsqueeze(-1)
        return x * self.act(x2).expand_as(x)


if(__name__ == "__main__"):
    block = ECABlock(32)
    x = torch.randn((1,32,16,16))
    y = block(x)
    print(y.shape)