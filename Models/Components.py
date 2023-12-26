import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF

class CBR(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(CBR, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unit(x)
    
class DCBR(nn.Module):
    """空洞卷积块"""
    def __init__(self, in_channels, out_channels, huge_kernel=True, use_GELU=False, dl=2) -> None:
        super(DCBR, self).__init__()
        self.kernel_size = 5 if(huge_kernel) else 3
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=(dl*(self.kernel_size-1)+1)//2, dilation=dl, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU() if(use_GELU) else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unit(x)

class BRC(nn.Module):
    """倒置卷积块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(BRC, self).__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.unit(x)

class D_CBR(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(D_CBR, self).__init__()
        self.unit = nn.Sequential(
            CBR(in_channels, out_channels),
            CBR(out_channels, out_channels)
        )

    def forward(self, x):
        return self.unit(x)

class D_BRC(nn.Module):
    """双倒置卷积块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(D_BRC, self).__init__()
        self.unit = nn.Sequential(
            BRC(in_channels, out_channels),
            BRC(out_channels, out_channels)
        )

    def forward(self, x):
        return self.unit(x)

class T_DCBR(nn.Module):
    """三空洞卷积块"""
    def __init__(self, in_channels, out_channels, dls=[2,3,5]) -> None:
        super(T_DCBR, self).__init__()
        assert len(dls) == 3, "长度非法"
        self.unit = nn.Sequential(
            DCBR(in_channels, out_channels, dl=dls[0]),
            DCBR(out_channels, out_channels, huge_kernel=False, dl=dls[1]),
            DCBR(out_channels, out_channels, huge_kernel=False, dl=dls[2])
        )

    def forward(self, x):
        return self.unit(x)

class UpSampling(nn.Module):
    """反卷积上采样模块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(UpSampling, self).__init__()
        self.opt = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        return self.opt(x)

class BilinearInterpolation(nn.Module):
    """双线性插值上采样模块"""
    def __init__(self, in_channels, out_channels, scale=2, mode="bilinear", align_corners=True):
        super(BilinearInterpolation, self).__init__()
        self.opt = nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.opt(x))

class ECABlock(nn.Module):
    """高效通道注意力机制模块"""
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

class SABlock(nn.Module):
    """空间注意力机制模块"""
    def __init__(self, huge_kernel=True):
        super(SABlock, self).__init__()
        self.kernel_size = 7 if(huge_kernel) else 3
        self.conv = nn.Conv2d(2, 1, self.kernel_size, padding=self.kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        marx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([mean, marx], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class DSA(nn.Module):
    """膨胀空间注意力"""
    def __init__(self, huge_kernel=True, dls=[2,3,5]):
        super(DSA, self).__init__()
        self.kernel_size = 5 if(huge_kernel) else 3
        self.dconvs = nn.ModuleList([nn.Conv2d(2, 1, self.kernel_size, \
            padding=(i*(self.kernel_size-1)+1)//2, dilation=i, bias=False) for i in dls])
        self.dcbre = D_CBR_ECA(len(dls), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        marx, _ = torch.max(x, dim=1, keepdim=True)
        x, cc = torch.cat([mean, marx], dim=1), None
        for module in self.dconvs:
            if(cc is None): cc = module(x)
            else: cc = torch.cat([cc, module(x)], dim=1)
        return self.sigmoid(self.dcbre(cc))

class T2DSA(DSA):

    def __init__(self, huge_kernel=True, dls=[2,3,5], inter=8):
        super(T2DSA, self).__init__(huge_kernel, dls)
        self.dconvs = nn.ModuleList([nn.Conv2d(2 if(n == 0) else inter, inter, self.kernel_size, \
            padding=(i*(self.kernel_size-1)+1)//2, dilation=i, bias=False) for n, i in enumerate(dls)])
        self.dcbre = D_CBR_ECA(inter, 1)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        marx, _ = torch.max(x, dim=1, keepdim=True)
        x= torch.cat([mean, marx], dim=1)
        for module in self.dconvs: x = module(x)
        return self.sigmoid(self.dcbre(x))

class D_CBR_ECA(nn.Module):
    """双卷积ECA模块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(D_CBR_ECA, self).__init__()
        self.unit = D_CBR(in_channels, out_channels)
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        return self.eca(self.unit(x))

class Mish(nn.Module):
    """Mish激活"""
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class MRes_ECA(nn.Module):
    """Mish残差ECA编码平行卷积块"""
    def __init__(self, in_channels, out_channels):
        super(MRes_ECA, self).__init__()
        self.inc = in_channels
        self.rec_in = in_channels if(out_channels == 2 * in_channels) else out_channels // 2
        self.init = nn.Conv2d(in_channels, self.rec_in, 7, 1, 3)
        self.primary = nn.Sequential(
            nn.Conv2d(self.rec_in, self.rec_in, 3, 1, 1),
            nn.BatchNorm2d(self.rec_in),
            Mish(),
            nn.Conv2d(self.rec_in, self.rec_in, 3, 1, 1),
            nn.BatchNorm2d(self.rec_in),
            Mish()
        )
        self.side = nn.Sequential(
            nn.Conv2d(self.rec_in, self.rec_in, 1, 1, 0),
            nn.BatchNorm2d(self.rec_in)
        )
        self.out_mish = Mish()
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        if(self.inc != self.rec_in): x = self.init(x)
        pr = self.primary(x)
        sd = self.side(x)
        c = torch.concat([pr, sd], dim=1)
        c = self.out_mish(c)
        return self.eca(c)

class T_CBR_ECA(nn.Module):
    """三卷积ECA模块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(T_CBR_ECA, self).__init__()
        self.unit = nn.Sequential(
            CBR(in_channels, out_channels),
            D_CBR(out_channels, out_channels)
        )
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        return self.eca(self.unit(x))

class Q_CBR_ECA(nn.Module):
    """四卷积ECA模块"""
    def __init__(self, in_channels, out_channels) -> None:
        super(Q_CBR_ECA, self).__init__()
        self.unit = nn.Sequential(
            D_CBR(in_channels, out_channels),
            D_CBR(out_channels, out_channels)
        )
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        return self.eca(self.unit(x))
    
class Q_DCBR_ECA(nn.Module):
    """四空洞卷积ECA"""
    def __init__(self, in_channels, out_channels) -> None:
        super(Q_DCBR_ECA, self).__init__()
        self.unit = nn.Sequential(
            CBR(in_channels, out_channels),
            T_DCBR(out_channels, out_channels)
        )
        self.eca = ECABlock(out_channels)

    def forward(self, x):
        return self.eca(self.unit(x))

class UAttentionGate(nn.Module):
    """UNet注意力门"""
    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(UAttentionGate, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor)
        self.g = nn.Sequential(
            nn.Conv2d(down_from_channel, down_to_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(down_to_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(down_to_channel, uni_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(uni_channel)
        )
        self.x = nn.Sequential(
            nn.Conv2d(left_channel, uni_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(uni_channel)
        )
        self.act = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(uni_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        ) 

    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        p = self.psi(self.act(d + l))
        return left * p
    
class DSAG(UAttentionGate):
    """空洞空间注意力门"""
    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(DSAG, self).__init__(down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor)
        self.dsa = DSA()
        self.weighting = D_BRC(2, 1)
    
    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        s = self.dsa(left)
        p = self.psi(self.act(d + l))
        c = self.weighting(torch.cat([s, p], dim=1))
        return left * c
    
class T2DSAG(DSAG):

    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(T2DSAG, self).__init__(down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor)
        self.weighting = D_BRC(left_channel * 2, left_channel)
    
    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        s = self.dsa(left)
        p = self.psi(self.act(d + l))
        return self.weighting(torch.cat([left * p, left * s], dim=1))
    
class T3DSAG(DSAG):

    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(T3DSAG, self).__init__(down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor)
        self.dsa = T2DSA()
        self.weighting = D_CBR_ECA(2, 1)
    
    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        s = self.dsa(left)
        p = self.psi(self.act(d + l))
        c = self.weighting(torch.cat([s, p], dim=1))
        return left * c
    
class T4DSAG(DSAG):

    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(T4DSAG, self).__init__(down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor)
        self.dsa = T2DSA(inter=32)
        self.weighting = D_CBR_ECA(left_channel * 2, left_channel)
    
    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        s = self.dsa(left)
        p = self.psi(self.act(d + l))
        return self.weighting(torch.cat([left * p, left * s], dim=1))

class T5DSAG(UAttentionGate):

    def __init__(self, down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor=2):
        super(T5DSAG, self).__init__(down_from_channel, down_to_channel, left_channel, uni_channel, scale_factor)
        self.dsa = T2DSA(inter=32)

    def forward(self, down, left):
        down = self.up(down)
        down = FF.resize(down, left.shape[-2:])
        d = self.g(down)
        l = self.x(left)
        s = self.dsa(left)
        p = self.psi(self.act(d + l + s))
        return left * p

class PAM(nn.Module):
    """位置注意力模块 CPC"""
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.in_channels = in_channels
        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.parameter.Parameter(torch.zeros(1))
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, height, width = x.size()
        proj_query = self.conv_q(x).view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv_k(x).view(batchsize, -1, width * height)
        attention = self.act(torch.bmm(proj_query, proj_key))
        proj_value = self.conv_v(x).view(batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return self.gamma * out.view(batchsize, C, height, width) + x

class SQPAM(nn.Module):

    def __init__(self, in_channels, sq=64):
        super(SQPAM, self).__init__()
        self.in_channels = in_channels
        self.pam = PAM(self.in_channels)
        self.squeeze = sq
    
    def forward(self, x):
        height, width = x.size()[-2:]
        if(height > self.squeeze and width > self.squeeze): 
            x = self.pam(FF.resize(x, [self.squeeze, self.squeeze])) 
            x = FF.resize(x, [height, width])
        else: x = self.pam(x)
        return x

class AmpleAttentionGate(nn.Module):
    """丰裕注意力门"""
    def __init__(self, left_channel:int, seq_channels:list=[], use_gelu=False):
        super(AmpleAttentionGate, self).__init__()

        if(left_channel in seq_channels): seq_channels.remove(left_channel)

        self.seq_channels = seq_channels
        uni_channel = left_channel // 2

        self.x = nn.Sequential(
            nn.Conv2d(left_channel, uni_channel, 1, 1, 0),
            nn.BatchNorm2d(uni_channel)
        )

        self.gs = nn.Sequential()
        for seq in seq_channels:
            self.gs.append(nn.Sequential(
                nn.Conv2d(seq, uni_channel, 1, 1, 0),
                nn.BatchNorm2d(uni_channel)
            ))

        self.act = nn.GELU() if(use_gelu) else nn.ReLU(inplace=True)

        self.uni = nn.Sequential(
            nn.Conv2d(uni_channel * (len(seq_channels) + 1), 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, left, *seqs):
        assert len(seqs) == len(self.seq_channels), "前向 通道顺序异常"
        lx = self.x(left)
        seqs = [FF.resize(seq, left.shape[-2:]) for seq in seqs]
        for i, g in enumerate(self.gs): seqs[i] = g(seqs[i])
        unied = self.uni(self.act(torch.cat(seqs + [lx], dim=1)))
        return left * unied

class STX(nn.Module):
    """直通块 用以替换"""
    def __init__(self) -> None:
        super(STX, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

if(__name__ == "__main__"):
    import numpy as np
    data = torch.tensor(np.ones((4, 32, 512, 512), dtype=np.float32))
    pam = SQPAM(32)
    out = pam(data)
    print(out.shape)

