from collections import OrderedDict
from typing import Dict
from torch import Tensor
from torchvision.models import vgg16_bn
from UNetSEG.Models.DV.unete import Up,OutConv
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)

        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Sigmoid()
             )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class SCAB(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SCAB, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch//2,kernel_size=1,stride=1),
            nn.ReLU(out_ch//2),
            nn.Conv2d(out_ch//2,out_ch,kernel_size=1,stride=1),

        )
        self.act = nn.Sigmoid()

    def forward(self,x):
        up = self.conv(x)
        return up, self.act(up)



# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 利用1x1卷积代替全连接
#         self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
#
# class cbam_block(nn.Module):
#     def __init__(self, channel, ratio=8):
#         super(cbam_block, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio=ratio)
#
#
#     def forward(self, x):
#         x = x * self.channelattention(x)
#         return x

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs(math.log(in_channel,2+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x





class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class VGG16UNet(nn.Module):

    model_name = "VGG16UNet"

    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(VGG16UNet, self).__init__()
        backbone = vgg16_bn(pretrained=pretrain_backbone)


        self.cbam_1 = ECA(64)
        self.cbam_2 = ECA(128)
        self.cbam_3 = ECA(256)
        self.cbam_4 = ECA(512)
        self.cbam_5 = ECA(512)

        #self.aspp_=ASPP(512,(2,6,10),512)
        # 全局平均池化,不管特征图多大,最后的输出结果是1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sca = SCAB(512, 512)
        self.sca1 = SCAB(512, 256)
        self.sca2 = SCAB(256, 128)
        self.sca3 = SCAB(128, 64)
        # 载入vgg16_bn预训练权重
        # https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
        #backbone.load_state_dict(torch.load(r"D:\deep_learning\AlexNet\unet\src\vgg16_bn-6c64b313.pth", map_location='cuda'))

        backbone = backbone.features

        stage_indices = [5, 12, 22, 32, 42]
        self.stage_out_channels = [64, 128, 256, 512, 512]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)


        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])

        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        backbone_out = self.backbone(x)

        #加入CA
        backbone_out['stage0'] = self.cbam_1(backbone_out['stage0'])
        backbone_out['stage1'] = self.cbam_2(backbone_out['stage1'])
        backbone_out['stage2'] = self.cbam_3(backbone_out['stage2'])
        backbone_out['stage3'] = self.cbam_4(backbone_out['stage3'])
        backbone_out['stage4'] = self.cbam_5(backbone_out['stage4'])


        #加入SCAB
        #aspp=self.aspp_(backbone_out['stage4'])

        glo_5=self.avg_pool(backbone_out['stage4'])

        glo_c4_up,glo_c4_dot = self.sca(glo_5)
        backbone_out['stage3'] = backbone_out['stage3'] * glo_c4_dot

        glo_c3_up,glo_c3_dot = self.sca1(glo_c4_up)
        backbone_out['stage2'] = backbone_out['stage2'] * glo_c3_dot

        glo_c2_up,glo_c2_dot= self.sca2(glo_c3_up)
        backbone_out['stage1'] = backbone_out['stage1'] * glo_c2_dot

        _,glo_c1_dot = self.sca3(glo_c2_up)
        backbone_out['stage0'] = backbone_out['stage0'] * glo_c1_dot

        #把backbone_out['stage4']换成aspp,那么aspp当做桥接进行上采样
        #不换成aspp，那么aspp经过global_avg再往上传递信息
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        #x = self.up1(aspp, backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)

        return {"out": x}

if __name__ == '__main__':
    print(VGG16UNet(2,False))
    # m = VGG16UNet(2,False)
    # # extract layer1 and layer3, giving as names `feat1` and feat2`
    # m=m.backbone
    # print(torchsummary.summary(m.cuda(),(3,333,333)))

    # new_m = IntermediateLayerGetter(m, {'5': 'stage0', '12': 'stage1', '22': 'stage2', '32': 'stage3','42':'stage4'})
    # out = new_m(torch.rand(1, 3, 512, 512))
    # print([(k, v.shape) for k, v in out.items()])
