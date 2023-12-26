from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Components import ECABlock


class CRGConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CRGConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.g1 = GhostConv(out_channels, out_channels)
        self.g2 = GhostConv(out_channels, out_channels)
        self.cv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.mi = nn.Mish()
        self.re = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):  # 没有进行下采样
        y = self.conv(x)
        # y = self.bn(y)


        # 卷积模块
        x = self.cv1(x)
        # x = self.bn(x)
        x = self.mi(x)
        # ghost模块
        x = self.g1(x)
        # x = self.bn(x)
        x = self.mi(x)
        # ghost模块
        x = self.g2(x)
        # x = self.bn(x)
        x = self.mi(x)
        # 卷积模块
        x = self.cv2(x)
        # x = self.bn(x)
        x = self.mi(x)

        # x = self.cv1(x)
        # x = self.bn(x)
        # x = self.re(x)
        #
        # x = self.cv2(x)
        # x = self.bn(x)
        # x = self.re(x)

        x += y

        # 残差模块
        return x

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class SiLU(nn.Module):
    
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # GhostConv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Mish(inplace=True),
            # nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # GhostConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
            # nn.ReLU(inplace=True)
        )


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()

# 上采样 + contact + 卷积
class Up(nn.Module):
    # bilinear 是否采用双线性插值替代转置卷积
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:  # 使用双线性差值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # mid_channels需要减半，因为第一个卷积之后通道要减半
        else:  # 使用转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 这里防止上采样之后拼接时两者尺寸不一致（预防图片的输入不是16的整数倍）
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 这样拼接的时候x1与x2的高宽才会一致

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 最后一个1x1的卷积
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,  # 经验证采用双线性差值和转置卷积效果是差不多的，但是双效率更高
                 base_c: int = 64): # 第一个卷积层传入卷积核的个数
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.eca1 = ECABlock(base_c)
        self.eca2 = ECABlock(base_c*2)
        self.eca3 = ECABlock(base_c*4)
        self.eca4 = ECABlock(base_c*8)

        self.pool = nn.MaxPool2d(2, stride=2)

        # Down里面包含了DoubleConv
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = CRGConv(base_c, base_c * 2)
        self.down2 = CRGConv(base_c * 2, base_c * 4)
        self.down3 = CRGConv(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1   # 使用双线性差值的话就要将通过除以二
        self.down4 = CRGConv(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x1 = self.in_conv(x)
        x1 = self.eca1(x1)

        a = self.pool(x1)
        x2 = self.down1(a)
        x2 = self.eca2(x2)

        b = self.pool(x2)
        x3 = self.down2(b)
        x3 = self.eca3(x3)

        c = self.pool(x3)
        x4 = self.down3(c)
        x4 = self.eca4(x4)

        d = self.pool(x4)
        x5 = self.down4(d)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}


if __name__ == '__main__':
    net = UNet(in_channels=3, num_classes=2).cuda()
    summary(net, (3, 512, 512))
    # with open('./text.txt', "w") as f:
    #     f.write("1234")

    # net(torch.zeros((1,3,512,512)))
    # out = net(torch.zeros((1,3, 512, 512)))

