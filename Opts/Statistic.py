from UNetSEG.Utils.UtilsStatistic import summary, monitor

# 引入模型类
from UNetSEG.Models.UNet import UNet
from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
from UNetSEG.Models.UPPCES import UNetPPCTRECASA
# from UNetSEG.Models.UPPRCES import UPPRCES
from UNetSEG.Models.UPPDR34CES import UPPDR34CES
from UNetSEG.Models.DPUPP import DPUPP
from UNetSEG.Models.DeepLab.deeplabv3 import DeeplabV3
from UNetSEG.Models.UPPAQCES import UPPAQCES
from UNetSEG.Models.TransUNet.vit_seg_modeling import VisionTransformer
from UNetSEG.Models.TransUNet.vit_seg_configs import get_r50_b16_config
from UNetSEG.Models.SegNet import SegNet


"""统计 - 统计指定模型的参数量"""
# model = VisionTransformer(get_r50_b16_config(), 512)
model = DPUPP(3, 2, 5)
# model = SegNet(3, 2)


monitor(summary(model.cuda(), (3, 64, 64)))