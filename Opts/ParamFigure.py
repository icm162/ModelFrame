import numpy as np
import tqdm

from matplotlib import pyplot as plt

from UNetSEG.Utils.UtilsStatistic import summary, monitor

# 引入模型类
from UNetSEG.Models.DPUPP import DPUPP
from UNetSEG.Models.UPPAQCES import UPPAQCES
from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
from UNetSEG.Models.SegNet import SegNet
from UNetSEG.Models.TransUNet.vit_seg_modeling import VisionTransformer, CONFIGS
from UNetSEG.Models.TransAttUNet.TransAttUnet import UNet_Attention_Transformer_Multiscale as TransAttUNet
from UNetSEG.Models.DeepLab.deeplabv3 import DeepLabV3

coord_lim = 0.8
alpha = 0.5
x_sp = 0.03
y_sp = 0.3

font = { "family": "Times New Roman", "size":11 }
bold_font = { "family": "Times New Roman", "size":11, "weight":"bold" }

o_features = [32 * 2**i for i in range(5)]
features = [16 * 2**i for i in range(6)]
mps = {
    SegNet: (3, 2),
    UNetPlusPlusCTR: (3, 2, o_features),
    VisionTransformer: (CONFIGS.get("R50-ViT-B_16"), 256, 2),
    TransAttUNet: (3, 2),
    DeepLabV3: (2, 16),
    UPPAQCES: (3, 2, features),
    DPUPP: (3, 2, 5)
}

vs = {
    SegNet: {"GlaS": 89.32, "CRAG": 90.73},
    UNetPlusPlusCTR: {"GlaS": 91.17, "CRAG": 90.52},
    VisionTransformer: {"GlaS": 91.53, "CRAG": 92.50},
    TransAttUNet: {"GlaS": 91.02, "CRAG": 91.17},
    DeepLabV3: {"GlaS": 92.34, "CRAG": 91.69},
    UPPAQCES: {"GlaS": 92.92, "CRAG": 91.56},
    DPUPP: {"GlaS": 92.78, "CRAG": 92.79}
}

def count_params(model):
    return sum(param.numel() for param in model.parameters() if(param.requires_grad))

bar = tqdm.tqdm(mps.items())
params_map = {}

for model_class, params in bar:
    model = model_class(*params)
    params_map[model_class.model_name] = (count_params(model), *vs.get(model_class).values())
    # monitor(summary(model.cuda(), (3, 128, 128)))

for name, params in params_map.items():
    print(f"{name}: {params}")

names = ["SegNet", "U-Net++", "TransUNet", "TransAttUNet", "DeepLabV3", "UPPAQCES", "DP-U-Net++"]
sizes, Y_GlaS, X_CRAG = zip(*params_map.values())
names = [n + f"({round(sizes[i] / 1e6, 2)}M)" for i, n in enumerate(names)]
sizes = list(map(lambda s: s / 70000, sizes))
colors = np.arange(1 / len(sizes) - 0.001, 1, 1 / len(sizes))
alphas = np.full(len(sizes), alpha)
uppers = [0, 5]
bolds = [5, 6]
offsets = [0,0.05,0.35,0.15,0.15,0,0.1]

plt.xlabel("dice coef on CRAG", fontdict=bold_font)
plt.ylabel("dice coef on GlaS", fontdict=bold_font)
plt.xticks(fontproperties=font.get("family"))
plt.yticks(fontproperties=font.get("family"))
plt.xlim(min(X_CRAG) - coord_lim, max(X_CRAG) + coord_lim)
plt.ylim(min(Y_GlaS) - coord_lim, max(Y_GlaS) + coord_lim)
for i, name in enumerate(names):
    plt.text(X_CRAG[i] - len(name) * x_sp, Y_GlaS[i] + (1 if(i in uppers) else -1) * (y_sp + offsets[i]), name, fontdict=(bold_font if(i in bolds) else font))
plt.scatter(x=X_CRAG, y=Y_GlaS, s=sizes, c=colors, alpha=alphas)
plt.show()



