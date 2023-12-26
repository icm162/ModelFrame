import torch
import torch.nn as nn

import torchvision as tv
import torchvision.transforms.functional as F
from UNetSEG.Models.Components import MRes_ECA, D_BRC, BRC, BilinearInterpolation, SABlock

class UPPRCES(nn.Module):

    model_name = "U++RCES"

    def __init__(self, in_channels=3, num_classes=2, features=[32, 64, 128, 256], reveal=False, reveal_class=1) -> None:
        super(UPPRCES, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.reveal = reveal
        self.reveal_class = reveal_class
        
        # 下采样共有操作
        self.downward = nn.MaxPool2d(2)
        # 编解码器连接层
        self.junction = MRes_ECA(features[-1], 2 * features[-1])

        # 编码器平行卷积块
        self.encoders = nn.ModuleList()
        for i, f in enumerate(features): self.encoders.append(MRes_ECA(in_channels if(i == 0) else features[i - 1], f))

        # 解码器平行卷积块
        self.decoders = nn.ModuleList()
        for i, f in enumerate(reversed(features)): self.decoders.append(D_BRC(f * (i + 2), f))

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

        # 多端输出卷积
        self.supers = nn.ModuleList()  
        for _ in range(len(features)): self.supers.append(BRC(features[0], self.num_classes))

        # 拼接输出 ECA
        self.sa = SABlock()
        self.out_dcbr = MRes_ECA(len(features) * num_classes, num_classes)
        self.out_dbrc = D_BRC(num_classes + 1, num_classes)
        

    def forward(self, x):
        encoded, sloped= [], []
        # 空间注意力输出
        sa = self.sa(x)
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
        supered = [self.supers[i](sloped[i + 1][0]) for i in range(len(self.supers))]
        if(self.reveal):
            for no, super in enumerate(supered): 
                tv.utils.save_image(super[0, self.reveal_class, ...], f"../Reveal/{self.model_name}/supervise-{no + 1}.jpg")
        catout = torch.cat(supered, dim=1)
        ecaout = self.out_dcbr(catout)
        if(self.reveal): 
            tv.utils.save_image(ecaout[0, self.reveal_class, ...], f"../Reveal/{self.model_name}/ecaout.jpg")
            tv.utils.save_image(sa[0, ...], f"../Reveal/{self.model_name}/sa.jpg")
        catlast = torch.cat([ecaout, sa], dim=1)
        # 输出层
        return self.out_dbrc(catlast)
        
        
if(__name__ == "__main__"):
    """模型测试"""
    x = torch.randn((1, 3, 119, 257))
    model = UPPRCES(features=[32, 64, 128, 256])
    outs = model(x)
    print(outs.shape)


