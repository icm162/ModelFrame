import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
import pyselflib.file_find as ff
from torchvision.transforms import functional as F

class AiridDataset(Dataset):

    # 这些玩意是静态的

    # 文件后缀名过滤
    format_filter = r"mat"
    # 文件名前缀过滤
    name_prefix_filter = r"WiFiRxB200"
    # 文件中数据键名
    find = "previous_matrix"
    # 窗口大小
    window_size = 1024
    # 调试开关
    debug = True
    # 数据集划分
    split = [8, 2]

    def __init__(self, base_dir, sub="train") -> None:

        super(AiridDataset, self).__init__()

        self.base_dir = base_dir
        self.sub = sub

        file_list = ff.find_files(self.base_dir)
        if (len(self.format_filter) != 0): file_list = list(filter(lambda p: p.endswith(f'.{self.format_filter}'), file_list))
        if (len(self.name_prefix_filter) != 0): 
            file_list = list(filter(lambda p: p.split("\\")[-1].startswith(self.name_prefix_filter), file_list))

        self.data_X, self.data_Y = [], []
        for label, path in enumerate(file_list):
            data_dict = scio.loadmat(path)
            assert self.find in data_dict, "未能找到数据矩阵"
            mat_data = data_dict.get(self.find)
            assert mat_data.shape[0] == 1, "数据构形异常"
            if (mat_data.shape[1] < self.window_size): break
            self.data_X.append(np.concatenate((np.real(mat_data), np.imag(mat_data))))
            self.data_Y.append(label)
        self.data_X, self.data_Y = np.array(self.data_X), np.array(self.data_Y)

        if(self.debug):
            print(self.data_X.shape)
            print(self.data_Y.shape)

    def __len__(self):
        all_len = self.data_X.shape[0] * (self.data_X.shape[2] - self.window_size + 1)  
        tail = all_len % sum(self.split) 
        return int((all_len - tail) * self.split[0 if(self.sub == "train") else 1] / sum(self.split)) \
              + (min(self.split[0], tail) if(self.sub == "train") else max(0, tail - self.split[0]))

    def __getitem__(self, index):
        index = self.distrib(index)
        lent = (self.data_X.shape[2] - self.window_size + 1)
        no, fronm = index // lent, index % lent
        data, label = self.data_X[no, :, fronm:fronm+self.window_size], self.data_Y[no]
        data, label = F.to_tensor(data), torch.as_tensor(label, dtype=torch.int32)
        return torch.squeeze(data).to(torch.float32), label
    
    def distrib(self, index):
        assert len(self.split) == 2, "数据集划分比例有误"
        summ, no = sum(self.split), 0 if(self.sub == "train") else 1 if(self.sub == "val") else -1
        return (index // self.split[no]) * summ + (index % self.split[no]) + (0 if(self.sub == "train") else self.split[0])


if(__name__ == "__main__"):
    # 数据基础目录
    base_dir = r"J:\AirID_Globecom2020_dataset\PhaseImpairedandNoImpairment_data"

    tr_dataset = AiridDataset(base_dir, sub="train")
    va_dataset = AiridDataset(base_dir, sub="val")

    print(f"数据长度为 {len(tr_dataset)}, {len(va_dataset)}")

    import random as r
    data_tuple = tr_dataset[r.randint(0, len(tr_dataset))]

    print(f"随机获取的数据为 {data_tuple[0].shape}  {data_tuple[1]}")
    print(data_tuple[0].dtype)
    
    for tp in tr_dataset:
        pass

    print("遍历测试完成")

