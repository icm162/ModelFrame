import torch
import torch.nn as nn
import torchsummary
import torchvision.transforms.functional as F
from UNetSEG.Models.Components import D_CBR, D_BRC, BRC, BilinearInterpolation

class UNetPlusPlus(nn.Module):

    model_name = "UNet++"

    def __init__(self, in_channels=3, num_classes=2, deep_supervision=True, features=[64, 128, 256, 512]) -> None:
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.features = features
        
        # 下采样共有操作
        self.downward = nn.MaxPool2d(2)
        # 编解码器连接层
        self.junction = D_CBR(features[-1], 2 * features[-1])

        # 编码器平行卷积块
        self.encoders = nn.ModuleList()
        for i, f in enumerate(features): self.encoders.append(D_CBR(in_channels if(i == 0) else features[i - 1], f))

        # 解码器平行卷积块
        self.decoders = nn.ModuleList()
        for i, f in enumerate(reversed(features)): self.decoders.append(D_BRC(f * (i + 2), f))

        # 多端损失
        if(self.deep_supervision):
            self.supers = nn.ModuleList()  
            for _ in range(len(features)): self.supers.append(BRC(features[0], self.num_classes))

        # 中间层填充卷积块  上层至下层
        self.conjs = nn.ModuleList()
        for i in range(len(features) - 1):
            layer_conjs = nn.ModuleList()
            for j in range(len(features) - 1 - i): layer_conjs.append(D_BRC(features[i] * (j + 2), features[i]))
            self.conjs.append(layer_conjs)

        # 上采样  左下层至右上层
        self.upwards = nn.ModuleList()
        for i in range(len(features)):
            slope_upward = nn.ModuleList()
            for j in range(len(features) - i): 
                slope_upward.append(BilinearInterpolation(features[j] * 2 if(j == len(features) - 1) else features[j + 1], features[j]))
            self.upwards.append(slope_upward)

    def forward(self, x):
        encoded, sloped= [], []
        # 编码器输出
        for i, m in enumerate(self.encoders): encoded.append(m(x) if(i == 0) else m(self.downward(encoded[i - 1])))
        # 桥接层输出
        junctioned = self.junction(self.downward(encoded[-1]))
        # 所有斜层 输出  左下至右上
        encoded.append(junctioned)
        sloped.append(encoded)
        for i in range(len(self.features)): sloped.append([])
        for i in range(len(self.features)):
            for j in range(len(self.features) - i):
                parallels = [s[j] for s in sloped[:i+1]]
                upward = self.upwards[i][j](sloped[i][j + 1])
                if(upward.shape[2:] != parallels[-1].shape[2:]): upward = F.resize(upward, parallels[-1].shape[2:])
                input = torch.cat([upward] + parallels, 1)
                sloped[i + 1].append(self.conjs[j][i](input) if(j != len(self.features) - i - 1) else self.decoders[i](input))
        # 输出层
        if(self.deep_supervision): return [self.supers[i](sloped[i + 1][0]) for i in range(len(self.supers))]
        else: return self.supers[-1](sloped[-1][0])
        
        
if(__name__ == "__main__"):
    """模型测试"""
    x = torch.randn((1, 3, 127, 345))
    model = UNetPlusPlus(features=[32, 64, 128, 256])

    #print(torchsummary.summary(model.cuda(), (3, 256, 256)))

    # print(model.upwards)
    outs = model(x)
    for o in outs: print(o.shape, end="  ")


