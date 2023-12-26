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
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
            trans.ToTensor()
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ToTensor()
        ]
    ),
    "test": trans.Compose(
        [
            trans.ToPILImage(),
            trans.ToTensor()
        ]
    )
}

class ChessBoardDataset(Dataset):

    data_name = "ChessBoard"

    def __init__(self, data_dir, sub="train", contour=False) -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub)
        self.datas = list(filter(lambda p: p.count("lbl") == 0, os.listdir(self.data_dir)))
        self.contour_focus = contour

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, self.datas[index]), cv.IMREAD_COLOR)
        if(self.sub != "test"): 
            label = cv.imread(os.path.join(self.data_dir, self.datas[index].replace(".jpg", "_lbl.jpg")
                                       .replace(".png", "_lbl.png")), cv.IMREAD_GRAYSCALE)
            if(self.contour_focus): contour = contour_focus(label)
            label = label // 255
            
        if(self.sub in transInfo):
            if(self.sub == "test"): data, _ = transInfo.get(self.sub)(data, data)
            elif(self.contour_focus): data, label, contour = transInfo.get(self.sub)(data, label, contour)
            else: data, label = transInfo.get(self.sub)(data, label)

        if(self.sub == "test"): return data, self.datas[index].replace(".jpg", "").replace(".png", "")
        elif(self.contour_focus): return data, label.to(torch.int64), contour.to(torch.int64)
        else: return data, label.to(torch.int64)

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = ChessBoardDataset(r"F:\DAT\ChessBoard", "train")
    data,label = dataset[25]
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