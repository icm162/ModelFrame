import torch
import torch.nn as nn

import torchvision as tv
import torchvision.transforms.functional as F
from UNetSEG.Models.Components import T_CBR_ECA, D_BRC, BRC, BilinearInterpolation, SABlock, UAttentionGate

class UPPATCES(nn.Module):

    model_name = "U++ATCES"

    def __init__(self, in_channels=3, num_classes=2, features=[32, 64, 128, 256, 512], reveal=False, reveal_class=1) -> None:
        super(UPPATCES, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.reveal = reveal
        self.reveal_class = reveal_class
        
        # 下采样共有操作
        self.downward = nn.MaxPool2d(2)

        # 结点列表 二维 左下至右上 左上至右下
        self.nodes = nn.ModuleList()
        for j in range(len(features)):
            reverse_slope = nn.ModuleList()
            for i in range(len(features) - j):
                reverse_slope.append(T_CBR_ECA(in_channels if(i == 0) else features[i - 1], features[i]) if(j == 0)
                                    else D_BRC(int((j + (1 if(i == 0) else 1.5)) * features[i]), features[i]))
            self.nodes.append(reverse_slope)

        # 跳联注意力门 二维 左下至右上 左上至右下
        self.ags = nn.ModuleList()
        for j in range(len(features) - 1): # 0 1 2 3
            layer_ags = nn.ModuleList()
            for i in range(len(features) - 1 - j): layer_ags.append(UAttentionGate(features[i + 1], features[i], features[i], features[i] // 2))
            self.ags.append(layer_ags)

        # 上采样 二维 左下至右上 左上至右下
        self.upwards = nn.ModuleList()
        for j in range(len(features) - 1):
            slope_upward = nn.ModuleList()
            for i in range(len(features) - 1 - j): slope_upward.append(BilinearInterpolation(features[i + 1], features[i]))
            self.upwards.append(slope_upward)

        # 多端输出卷积
        self.supers = nn.ModuleList()  
        for _ in range(len(features) - 1): self.supers.append(BRC(features[0], self.num_classes))

        # 拼接输出 ECA
        self.sa = SABlock()
        self.out_dcbr = T_CBR_ECA((len(features) - 1) * num_classes, num_classes)
        self.out_dbrc = D_BRC(num_classes + 1, num_classes)
        

    def forward(self, x):
        uouts, atos, val_nodes, ups = [], [[]] * (len(self.features) - 1), [], []
        # 空间注意力输出
        sa = self.sa(x)
        for i, node in enumerate(self.nodes[0]): val_nodes.append(node(x if(i == 0) else self.downward(val_nodes[-1])))
        for i, v in enumerate(val_nodes[1:]): ups.append(F.resize(self.upwards[0][i](v), val_nodes[i].shape[2:]))
        for i in range(len(self.features) - 1): atos[0].append(self.ags[0][i](val_nodes[i + 1], val_nodes[i]))
        for i in range(1, len(self.features) - 1):
            val_nodes = []
            for j in range(len(self.features) - i):
                inps = [atos[a][j] for a in range(i)] + [ups[j]] + ([self.downward(val_nodes[-1])] if(j != 0) else [])
                val_nodes.append(self.nodes[i][j](torch.cat(inps, dim=1)))
            uouts.append(val_nodes[0])
            ups = []
            for j, v in enumerate(val_nodes[1:]): ups.append(F.resize(self.upwards[i][j](v), val_nodes[j].shape[2:]))
            for j in range(len(self.features) - 1 - i): atos[i].append(self.ags[i][j](val_nodes[j + 1], val_nodes[j]))
        final_inps = [atos[a][0] for a in range(len(self.features) - 1)] + [ups[0]]
        uouts.append(self.nodes[len(self.features) - 1][0](torch.cat(final_inps, dim=1)))
        supered = [self.supers[i](uouts[i]) for i in range(len(self.supers))]
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
    model = UPPATCES(features=[32, 64, 128, 256, 512])
    outs = model(x)
    print(outs.shape)


