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
            trans.RandomCrop(192),
            trans.ToTensor(),
            trans.Normalize([0.48536915, 0.582447, 0.7565686], [0.1963908, 0.19264077, 0.19684035])
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ToTensor(),
            trans.Normalize([0.501968, 0.55371284, 0.74127513], [0.20995556, 0.19722079, 0.19760402])
        ]
    )
}

class PH2Dataset(Dataset):

    data_name = "PH2"

    def __init__(self, data_dir, sub="train") -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub)
        self.images = os.listdir(os.path.join(self.data_dir, "image"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, "image", self.images[index]), cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(self.data_dir, "label", "Y" + self.images[index][1:]), cv.IMREAD_GRAYSCALE)
        contour = contour_focus(label)

        label = label // 255

        if(self.sub in transInfo): data, label, contour = transInfo.get(self.sub)(data, label, contour)

        return data, label.to(torch.int64), contour.to(torch.int64)

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = PH2Dataset(r"F:\DAT\PH2", "val")
    data,label,contour = dataset[16]
    print(data.shape)
    print(label.shape)
    print(contour.shape)
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