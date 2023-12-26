import os
import torch
import cv2 as cv
import numpy as np

from torch.utils.data import Dataset

import UNetSEG.Datasets.Transforms as trans

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
            trans.Normalize([0.7839318, 0.51658475, 0.7863229], [0.1299742, 0.24623536, 0.16419558])
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ForceResize(512),
            trans.ToTensor(),
            trans.Normalize([0.8031925, 0.54459965, 0.81119287], [0.118215516, 0.25020266, 0.15214856])
        ]
    )
}

class GlaSTransDataset(Dataset):

    data_name = "GlaS"

    def __init__(self, data_dir, sub="train") -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub)
        self.datas = list(filter(lambda p: p.count("mask") == 0, os.listdir(self.data_dir)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, self.datas[index]), cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(self.data_dir, self.datas[index].replace(".bmp", "_mask.bmp")), cv.IMREAD_GRAYSCALE)
        contour = contour_focus(label)

        label = label // 255

        if(self.sub in transInfo): data, label, contour = transInfo.get(self.sub)(data, label, contour)

        return data, label.to(torch.int64), contour.to(torch.int64)

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = GlaSTransDataset(r"F:\DAT\GlaST", "train")
    data,label,contour = dataset[25]
    print(data.dtype, label.dtype)
    print(f"{torch.max(data)}  {torch.min(data)}")
    data,label = data.permute(1,2,0).numpy(), label.to(torch.uint8).numpy()
    contour = contour.to(torch.uint8).numpy()
    print(f"{np.max(data)}  {np.min(data)}")
    print(f"{np.max(label)}  {np.min(label)}")
    print(f"{data.shape}  {label.shape}")
    cv.imshow("data", data)
    cv.imshow("label", label)
    cv.imshow("contour", contour)
    cv.waitKey(0)
    cv.destroyAllWindows()