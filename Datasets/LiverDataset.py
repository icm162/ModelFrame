import os
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import torch
import numpy as np

import UNetSEG.Datasets.Transforms as trans

train_mean=[0.11092731, 0.11092731, 0.11092731]
train_std=[0.17790428, 0.17790428, 0.17790428]

val_mean=[0.09642766, 0.09642766, 0.09642766]
val_std=[0.18618211, 0.18618211, 0.18618211]

train_mean.reverse()
train_std.reverse()

val_mean.reverse()
val_std.reverse()

transInfo = {
    "train": trans.Compose(
        [
            trans.ToPILImage(),
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
            trans.ToTensor(),
            trans.Normalize(train_mean, train_std)
        ]
    ),
    "val": trans.Compose(
        [
            trans.ToTensor(),
            trans.Normalize(val_mean, val_std)
        ]
    )
}

class LiverDataset(Dataset):

    data_name = "Liver"

    def __init__(self, data_dir, sub="train") -> None:
        super().__init__()
        self.sub = sub
        self.data_dir = os.path.join(data_dir, sub)
        self.datas_path = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.datas_path) // 2

    def __getitem__(self, index):

        data = cv.imread(os.path.join(self.data_dir, f"{str(index).zfill(3)}.png"), cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(self.data_dir, f"{str(index).zfill(3)}_mask.png"), cv.IMREAD_GRAYSCALE)

        if(self.sub in transInfo): data, label = transInfo.get(self.sub)(data, label)

        label[label==255] = 1
        return data, label

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    dataset = LiverDataset(r"F:\DAT\Liver\data\liver", "train")
    data,label = dataset[57]
    print(f"{torch.max(data)}  {torch.min(data)}")
    data,label = np.uint8(data.permute(1,2,0).numpy()), np.uint8(label.numpy())
    cv.imshow("data", data)
    cv.imshow("label", label)
    cv.waitKey(0)
    cv.destroyAllWindows()