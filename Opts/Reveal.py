import torch

# 引入数据集类
from UNetSEG.Datasets.GlaSDataset import GlaSDataset
from UNetSEG.Datasets.LiverDataset import LiverDataset

# 引入模型类
# from UNetSEG.Models.UNet import UNet
# from UNetSEG.Models.UNetPP import UNetPlusPlus
# from UNetSEG.Models.UNetPPO import UnetPlusPlus as UNetPlusPlusOri
# from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
# from UNetSEG.Models.vgg_unet import VGG16UNet
# from UNetSEG.Models.UNetPPCTRECA import UNetPlusPlusCTRECA
from UNetSEG.Models.UPPCES import UNetPPCTRECASA

# 引入工具包
from UNetSEG.Utils.UtilsPrint import cprint
import UNetSEG.Utils.UtilsCheckPoint as cp

"""可视化 - 调用能够可视化的模型输出前向时指定层的输出"""

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
data_number = 3

num_classes = 2
model = model_class(3, num_classes, reveal=True)

if(__name__ == "__main__"):

    # CUDA 可用标志
    CUDA_FLAG = torch.cuda.is_available()
    cprint(f"\nCUDA {'' if(CUDA_FLAG) else '不'}可用\n", "info" if(CUDA_FLAG) else "error")
    model = model.cuda() if(CUDA_FLAG) else model

    # 加载数据集
    dataset = dataset_class(path_datasets, "val")

    # 读档
    model, epstr = cp.load_model(model, path_saves, name_load)
    cprint(f"已读取 {str(epstr)} 轮训练参数\n", "healthy")

    with torch.no_grad():
        img, _ = dataset[data_number - 1]
        if(CUDA_FLAG): img = img.cuda()
        img = img.unsqueeze(0)
        model(img)
