import torch
import torch.nn as nn
from torchvision import models
from UNetSEG.Models.UPPAQCES import UPPAQCES

model = models.resnet101(pretrained=False)
# model = models.resnet50()
# model = models.resnet34()
# s_model = UPPAQCES()

model.layer1[-1].conv3 = nn.Conv2d(64, 128, 1, 1, 0, bias=False)


print(model)


# print(model)
# print(s_model.nodes)

# layer = model.layer1

# inp = torch.randn((1, 64, 13, 15))

# out = layer(inp)

# print(out.shape)


