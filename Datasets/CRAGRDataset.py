import os
import cv2 as cv
from torch.utils.data import Dataset
import UNetSEG.Datasets.Transforms as trans
import torch
import numpy as np

from UNetSEG.Utils.UtilsPreProcess import contour_focus

transInfo = {
    "train": trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomResize(int(0.5 * 565), int(1.2 * 565)),
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
            trans.RandomCrop(512),
            trans.ToTensor(),
            trans.Normalize([0.8542532, 0.7151299, 0.8267304], [0.099837795, 0.17074603, 0.13391654])
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ToTensor(),
            trans.Normalize([0.8598447, 0.7281379, 0.84246415], [0.09380259, 0.16527346, 0.122175425])
        ]
    )
}

class CRAGRDataset(Dataset):

    data_name = "CRAGR"

    def __init__(self, data_dir, sub="train", contour:bool=False) -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub)
        self.datas = os.listdir(os.path.join(self.data_dir, "image"))
        self.contour_focus = contour

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, "image", self.datas[index]), cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(self.data_dir, "label", self.datas[index]), cv.IMREAD_GRAYSCALE)
        if(self.contour_focus): contour = contour_focus(label)

        if(self.sub in transInfo): 
            if(self.contour_focus): data, label, contour = transInfo.get(self.sub)(data, label, contour)
            else: data, label = transInfo.get(self.sub)(data, label)

        label = label // 255

        if(self.contour_focus): return data, label.to(torch.int64), contour.to(torch.int64)
        else: return data, label.to(torch.int64)

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = CRAGRDataset(r"F:\DAT\CRAGR", "train")
    data,label = dataset[22]
    print(data.dtype, label.dtype)
    print(f"{torch.max(data)}  {torch.min(data)}")
    data,label = data.permute(1,2,0).numpy(), label.to(torch.uint8).numpy()
    print(f"{np.max(data)}  {np.min(data)}")
    print(f"{np.max(label)}  {np.min(label)}")
    print(f"{data.shape}  {label.shape}")
    cv.imshow("data", data)
    cv.imshow("label", label)
    cv.waitKey(0)
    cv.destroyAllWindows()