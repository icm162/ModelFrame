import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import UNetSEG.Datasets.Transforms as trans
import cv2 as cv
import torch

transInfo = {
    "train": trans.Compose(
        [
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
            trans.RandomCrop(512),
            trans.ToTensor(),
            trans.Normalize([0.7839318, 0.51658475, 0.7863229], [0.1299742, 0.24623536, 0.16419558])
            # trans.Normalize([0.8031925, 0.54459965, 0.81119287], [0.118215516, 0.25020266, 0.15214856])
        ]
    )
}

class GlaSUCDataset(Dataset):

    data_name = "GlaSUC"

    def __init__(self, root:str, sub:str):
        super(GlaSUCDataset, self).__init__()
        data_root = os.path.join(root, sub)
        assert os.path.exists(data_root), f"路径 {data_root} 不存在"
        self.transforms = transInfo[sub]
        img_names = [i for i in os.listdir(data_root) if i.endswith(".bmp")]
        self.img_list = list(map(lambda p: os.path.join(data_root, p), [i for i in img_names if(i.count("mask") == 0)]))
        self.manual = list(map(lambda p: p.replace(".bmp", "_mask.bmp"), self.img_list))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        mask = np.clip(manual, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if(__name__ == "__main__"):
    """测试数据集是否能正常读取"""
    mean = (0.7875132, 0.5128235, 0.7842114)
    std = (0.16708419, 0.2465137, 0.13172045)
    dataset = GlaSUCDataset(r"F:\DAT\GlaST", "train")
    img, mask = dataset[23]
    print("dtype", img.dtype, mask.dtype)
    ai, am = img.permute(1,2,0).numpy(), mask.to(torch.uint8).numpy()
    print(ai.shape, am.shape)
    print(np.max(ai), np.max(am))
    print(np.min(ai), np.min(am))
    img = np.transpose(img, (1,2,0))
    usePIL = True
    if(usePIL):
        plt.figure("data")
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.imshow(mask, cmap="gray")
        plt.show()
    else:
        cv.imshow("data", ai)
        cv.imshow("mask", am)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    
