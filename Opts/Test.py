import torchvision as tv
import torch
import tqdm
import os

from torch.utils.data import DataLoader

# 引入数据集类
from UNetSEG.Datasets.GlaSDataset import GlaSDataset
# from UNetSEG.Datasets.LiverDataset import LiverDataset
# from UNetSEG.Datasets.CRAGCDataset import CRAGCDataset
# from UNetSEG.Datasets.PH2Dataset import PH2Dataset

# 引入模型类
from UNetSEG.Models.UNet import UNet
from UNetSEG.Models.SegNet import SegNet
# from UNetSEG.Models.UNetPP import UNetPlusPlus
# from UNetSEG.Models.UNetPPO import UnetPlusPlus as UNetPlusPlusOri
from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
# from UNetSEG.Models.vgg_unet import VGG16UNet
# from UNetSEG.Models.UNetPPCTRECA import UNetPlusPlusCTRECA
# from UNetSEG.Models.UPPCES import UNetPPCTRECASA
# from UNetSEG.Models.UPPCDR34CES import UPPDR34CES

# 引入工具包
from UNetSEG.Utils.UtilsPrint import cprint
import UNetSEG.Utils.UtilsCheckPoint as cp

"""可视化 - 调用能够可视化的模型打印前向时指定层的输出"""

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
path_saves = r"J:\UNS\UPPCTR\UNet++Concentration-GlaS-L6B4E200DF"
path_predict = r"../Predict/"
name_load = r"UNet++Concentration-GlaS-L6B4E200DF-EP200-FINAL.pth"
dataset_class = GlaSDataset
model_class = UNetPlusPlusCTR

num_classes = 2
model = model_class(3, num_classes, features=[16, 32, 64, 128, 256, 512])

if(__name__ == "__main__"):

    # CUDA 可用标志
    CUDA_FLAG = torch.cuda.is_available()
    cprint(f"\nCUDA {'' if(CUDA_FLAG) else '不'}可用\n", "info" if(CUDA_FLAG) else "error")
    model = model.cuda() if(CUDA_FLAG) else model

    # 加载数据集
    ts_data = dataset_class(path_datasets, "val")
    ts_loader = DataLoader(ts_data, batch_size=1)
    tqdm_ts = tqdm.tqdm(ts_loader)

    # 读档
    model, epstr = cp.load_model(model, path_saves, name_load)
    cprint(f"已读取 {str(epstr)} 轮训练参数\n", "healthy")

    # 创建或清空模型对应预测文件夹
    predict_dir = os.path.join(path_predict, model_class.model_name)
    if(not os.path.exists(predict_dir)): os.mkdir(predict_dir)
    else: 
        for f in os.listdir(predict_dir): os.remove(os.path.join(predict_dir, f))

    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm_ts):
            if(CUDA_FLAG): img = img.cuda()
            out = model(img)
            rout = out["out"] if(isinstance(out, dict)) else out
            predict = torch.sigmoid(rout).cpu()
            tv.utils.save_image(predict[0, ...].argmax(0).to(torch.float16), os.path.join(predict_dir, f"val-{i + 1}.png"))

