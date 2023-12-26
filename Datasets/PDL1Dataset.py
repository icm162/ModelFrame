import os
import torch
import cv2 as cv
import numpy as np

from torch.utils.data import Dataset

import UNetSEG.Datasets.Transforms as trans

from UNetSEG.Utils.UtilsPreProcess import contour_focus


softint = lambda s: int(s) if s.isdigit() else 1


transInfo = {
    "train": trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomResize(int(0.5 * 565), int(1.2 * 565)),
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
            trans.RandomCrop(512),
            trans.ToTensor()
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ToTensor()
        ]
    )
}


class PDL1Dataset(Dataset):

    data_name = "PDL1"

    def __init__(self, data_dir, sub="train", contour=False) -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub, "image")
        self.datas = os.listdir(self.data_dir)
        sorted(self.datas, key=lambda n: (softint(n.split("-")[1]), int(n.split(".")[0].split("-")[-1])))
        self.contour_focus = contour


    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, self.datas[index]), cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(self.data_dir.replace("image", "label"), self.datas[index]), cv.IMREAD_GRAYSCALE)
        
        if(self.contour_focus): contour = contour_focus(label)

        label = label // 255

        if(self.sub in transInfo): 
            if(self.contour_focus): data, label, contour = transInfo.get(self.sub)(data, label, contour)
            else: data, label = transInfo.get(self.sub)(data, label)

        if(self.contour_focus): return data, label.to(torch.int64), contour.to(torch.int64) 
        else: return data, label.to(torch.int64)


if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = PDL1Dataset(r"F:\DAT\PDL1", "train", True)
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