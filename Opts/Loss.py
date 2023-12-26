import os
import torch
import platform
import cv2 as cv
import numpy as np

from torch import nn

# 引入数据集类
# from UNetSEG.Datasets.LiverDataset import LiverDataset
from UNetSEG.Datasets.GlaSDataset import GlaSDataset
# from UNetSEG.Datasets.CRAGCDataset import CRAGCDataset

# 引入模型类
# from UNetSEG.Models.UNet import UNet
# from UNetSEG.Models.UNetPP import UNetPlusPlus
# from UNetSEG.Models.UNetPPO import UnetPlusPlus as UNetPlusPlusOri
# from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
# from UNetSEG.Models.vgg_unet import VGG16UNet
# from UNetSEG.Models.UNetPPCTRECA import UNetPlusPlusCTRECA
from UNetSEG.Models.UPPCES import UNetPPCTRECASA

# 引入工具包
from UNetSEG.Utils.UtilsStandards import build_target, dice_loss
from UNetSEG.Utils.UtilsWindowsMessage import send_windows_message
from UNetSEG.Utils.UtilsPrint import cprint
from UNetSEG.Utils.UtilsPreProcess import contour_focus
import UNetSEG.Utils.UtilsCheckPoint as cp

"""计算损失 - 通过指定数据计算损失"""

"""
    数据路径
    读档路径
    读档文件名
    数据集类
    模型类
    数据序号

    分割类别数
    模型初始化参数
"""       
path_datasets = r"F:\DAT\GlaST"
path_saves = r"F:\PTC\UNetSEG\Saves\U++CES-GlaS-2"
name_load = r"U++CES-GlaS-2-EP67-BESTEN-85.485.pth"
dataset_class = GlaSDataset
model_class = UNetPPCTRECASA
data_number = 29

num_classes = 2
model = model_class(3, num_classes)

if(__name__ == "__main__"):

    # CUDA 可用标志
    CUDA_FLAG = torch.cuda.is_available()
    cprint(f"\nCUDA {'' if(CUDA_FLAG) else '不'}可用\n", "info" if(CUDA_FLAG) else "error")
    model = model.cuda() if(CUDA_FLAG) else model

    # 加载数据集
    dataset = dataset_class(path_datasets, "train")

    # 损失函数
    loss_weight = torch.as_tensor([1.0, 2.0])
    loss_weight_2 = torch.as_tensor([0.3, 0.1])
    loss_weight = loss_weight.cuda() if(CUDA_FLAG) else loss_weight
    loss_weight_2 = loss_weight_2.cuda() if(CUDA_FLAG) else loss_weight_2
    CE_Loss = lambda x, gt, w, wt=1: nn.functional.cross_entropy(x, gt, weight=w, ignore_index=255) * wt
    DICE_Loss = dice_loss

    # 读档
    model, epstr = cp.load_model(model, path_saves, name_load)
    cprint(f"已读取 {str(epstr)} 轮训练参数\n", "healthy")

    with torch.no_grad():
        img, target = dataset[data_number - 1]

        inp = np.uint8((target * 255).numpy())
        print(inp.max(), inp.min())

        cv.imshow("inp", inp)
        cv.imshow("cf", contour_focus(inp))
        cv.waitKey(0)
        cv.destroyAllWindows()

        ctfcs = torch.tensor(contour_focus(inp), dtype=torch.int64)
        if(CUDA_FLAG): img, target, ctfcs = img.cuda(), target.cuda(), ctfcs.cuda()
        img, target, ctfcs = img.unsqueeze(0), target.unsqueeze(0), ctfcs.unsqueeze(0)

        # hypo = torch.stack([dimm, target], dim=1)
        # sort = torch.sort(hypo, dim=1, descending=False).indices.to(torch.float32)

        out = model(img)
        out = out["out"] if(isinstance(out, dict)) else out
        sout = torch.sigmoid(out)
        dice_target = build_target(target, num_classes)

        ce = CE_Loss(sout, target, loss_weight)
        contour = CE_Loss(sout, ctfcs, loss_weight_2, 0.3)
        dice = DICE_Loss(sout, dice_target, multiclass=True)
        losses = ce + dice + contour

    cprint(f"总损失 {losses.item()}", "info")
    cprint(f"交叉熵损失 {ce.item()}", "info")
    cprint(f"轮廓交叉熵损失 {contour.item()}", "info")
    cprint(f"DICE损失 {dice.item()}", "info")

    # 发送系统提示消息
    if(platform.system() == "Windows"):
        send_windows_message(f"{model_class.model_name}测试完成", f"总损失 {losses.item():.3f}\n交叉熵损失 {ce.item():.3f}\n轮廓交叉熵损失 {contour.item():.3f}\nDICE损失 {dice.item():.3f}")






