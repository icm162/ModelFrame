import torch
import torch.nn as nn

import torchvision as tv
import torchvision.transforms.functional as F
from UNetSEG.Models.Components import Q_CBR_ECA, D_BRC, BRC, BilinearInterpolation, SABlock, UAttentionGate, ECABlock

class UPPAR34CES(nn.Module):

    model_name = "UPPAR34CES"

    def __init__(self, in_channels=3, num_classes=2, layer=6, reveal=False, reveal_class=1) -> None:
        super(UPPAR34CES, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert layer >= 5, "层数无法容纳迁移R34"
        self.features = [2**(i + 10 - layer) for i in range(layer)] # [16, 32, 64, 128, 256, 512]
        self.use_sa = True
        self.reveal = reveal
        self.reveal_class = reveal_class
        self.o_model = tv.models.resnet34(pretrained=True)
        
        # 下采样共有操作
        self.downward = nn.MaxPool2d(2)

        # 结点列表 二维 左下至右上 左上至右下
        self.nodes = nn.ModuleList()
        for j in range(len(self.features)):
            reverse_slope = nn.ModuleList()
            for i in range(len(self.features) - j):
                reverse_slope.append(Q_CBR_ECA(in_channels if(i == 0) else self.features[i - 1], self.features[i]) if(j == 0)
                                    else D_BRC(int((j + (1 if(i == 0) else 1.5)) * self.features[i]), self.features[i]))
            self.nodes.append(reverse_slope)
        
        reverse_slope = nn.ModuleList()
        reverse_slope.append(nn.Sequential(Q_CBR_ECA(in_channels, self.features[0]), self.downward))
        for i in range(layer - 5): 
            reverse_slope.append(nn.Sequential(Q_CBR_ECA(self.features[i], self.features[i + 1]), self.downward))
        self.first_tranchannel = nn.Conv2d(self.features[-5], self.features[-4], 1, 1, 0)
        reverse_slope.append(nn.Sequential(self.o_model.layer1, ECABlock(self.features[-4]), self.downward))
        reverse_slope.append(nn.Sequential(self.o_model.layer2, ECABlock(self.features[-3])))
        reverse_slope.append(nn.Sequential(self.o_model.layer3, ECABlock(self.features[-2])))
        reverse_slope.append(nn.Sequential(self.o_model.layer4, ECABlock(self.features[-1])))
        self.nodes[0] = reverse_slope

        # 跳联注意力门 二维 左下至右上 左上至右下
        self.ags = nn.ModuleList()
        for j in range(len(self.features) - 1):
            layer_ags = nn.ModuleList()
            for i in range(len(self.features) - 1 - j): layer_ags.append(UAttentionGate(self.features[i + 1], self.features[i], self.features[i], self.features[i] // 2))
            self.ags.append(layer_ags)

        # 上采样 二维 左下至右上 左上至右下
        self.upwards = nn.ModuleList()
        for j in range(len(self.features) - 1):
            slope_upward = nn.ModuleList()
            for i in range(len(self.features) - 1 - j): slope_upward.append(BilinearInterpolation(self.features[i + 1], self.features[i]))
            self.upwards.append(slope_upward)

        # 多端输出卷积
        self.supers = nn.ModuleList()  
        for _ in range(len(self.features) - 1): self.supers.append(BRC(self.features[0], self.num_classes))

        # 拼接输出 ECA
        if(self.use_sa): self.sa = SABlock()
        self.out_dcbr = Q_CBR_ECA((len(self.features) - 1) * num_classes, num_classes)
        self.out_dbrc = D_BRC(num_classes + 1 if(self.use_sa) else num_classes, num_classes)
        

    global div2
    div2 = lambda l: list(map(lambda i: i // 2, l))


    def forward(self, x):
        uouts, atos, val_nodes, ups = [], [[]] * (len(self.features) - 1), [], []
        # 空间注意力输出
        if(self.use_sa): sa = self.sa(x)

        for i, node in enumerate(self.nodes[0]): 
            if(i == 0): val_nodes.append(node(x))
            elif(i == len(self.features) - 4): 
                val_nodes.append(F.resize(node(self.first_tranchannel(val_nodes[-1])), div2(val_nodes[-1].shape[2:])))
            else: val_nodes.append(F.resize(node(val_nodes[-1]), div2(val_nodes[-1].shape[2:])))

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
        supered = [F.resize(s, x.shape[2:]) for s in supered]

        if(self.reveal):
            for no, super in enumerate(supered): 
                tv.utils.save_image(super[0, self.reveal_class, ...], f"../Reveal/{self.model_name}/supervise-{no + 1}.jpg")

        catout = torch.cat(supered, dim=1)
        ecaout = self.out_dcbr(catout)
        if(self.reveal): 
            tv.utils.save_image(ecaout[0, self.reveal_class, ...], f"../Reveal/{self.model_name}/ecaout.jpg")
            tv.utils.save_image(sa[0, ...], f"../Reveal/{self.model_name}/sa.jpg")
        sa = F.resize(sa, ecaout.shape[2:])
        catlast = torch.cat([ecaout, sa], dim=1) if(self.use_sa) else ecaout

        # 输出层
        return self.out_dbrc(catlast)
        
        
if(__name__ == "__main__"):
    """模型测试"""
    x = torch.randn((1, 3, 257, 343))
    model = UPPAR34CES(layer=6)
    outs = model(x)
    print(outs.shape)


