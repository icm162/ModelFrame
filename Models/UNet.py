import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchsummary

class ConvUnit(nn.Module):
    """一个 ConvUnit 是一个 UNet 结点块"""
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.unit(x)

class UNet(nn.Module):

    model_name = "UNet"

    def __init__(self, in_channels=3, num_classes=2, features=[64, 128, 256, 512]) -> None:
        super().__init__()
        self.upward = nn.ModuleList()
        self.right = nn.ModuleList()
        self.downward = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for chan in features:
            self.downward.append(ConvUnit(in_channels, chan))
            in_channels = chan

        for chan in reversed(features):
            self.upward.append(nn.ConvTranspose2d(chan * 2, chan, kernel_size=2, stride=2))

        for chan in reversed(features):
            self.right.append(ConvUnit(chan * 2, chan))

        self.conjunction = ConvUnit(features[-1], features[-1] * 2)

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        connections = []
        for down in self.downward:
            x = down(x)
            connections.append(x)
            x = self.pool(x)
        x = self.conjunction(x)
        connections = connections[::-1]
        for i in range(len(self.upward)):
            x = self.upward[i](x)
            connection = connections[i]
            if(x.shape != connection.shape): x = F.resize(x, connection.shape[2:])
            concat = torch.cat((connection, x), 1) # B'C'HW
            x = self.right[i](concat)
        return self.classifier(x)
        
if(__name__ == "__main__"):
    """模型测试"""
    x = torch.randn((1, 3, 451, 123))
    model = UNet()

    # print(torchsummary.summary(model.cuda(), (3, 256, 256)))

    outs = model(x)
    print(outs.shape)


