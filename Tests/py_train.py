import torchvision
from py_VGG import *
import torch
from torch.utils.data import DataLoader
from torch import nn
from Airidlearn.AiridDataset import AiridDataset

base_dir = r"E:\data\AirID_Globecom2020_dataset\PhaseImpairedandNoImpairment_data"

train_data = AiridDataset(base_dir, sub="train")
test_data = AiridDataset(base_dir, sub="val")

train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)

CUDA_FLAG = torch.cuda.is_available()
print(f"\nCUDA {'' if(CUDA_FLAG) else '不'}可用\n", "info" if(CUDA_FLAG) else "error")
if(not CUDA_FLAG): exit()

#搭建VGG网络
vgg = VGG().cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss().cuda()
#优化器
learning_rate = 0.01
optimizer = torch.optim.Adam(vgg.parameters(), lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0#记录测试的次数
total_test_step = 0#训练的轮数
epoch = 3


for i in range(epoch):
    print("---——--第{}轮训练开始-————--".format(i+1))
    # 训练步骤开始
    for data in train_dataloader:
        signals, targets = data
        signals, targets = signals.cuda(), targets.cuda()
        outputs = vgg(signals)
        #print(f"outputs {outputs.shape}   targets {targets.shape}")
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            precision = (1 - torch.count_nonzero(torch.argmax(outputs, dim=1) - targets) / targets.shape[0]) * 100
            print("训练次数:{}, Loss:{}, precision:{}%".format(total_train_step, loss.item(), precision))

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        all_nonzero, all = 0, 0
        for data in test_dataloader:
            signals, targets = data
            signals, targets = signals.cuda(), targets.cuda()
            outputs = vgg(signals)
            all_nonzero, all = all_nonzero + torch.count_nonzero(torch.argmax(outputs, dim=1) - targets), all + \
                               targets.shape[0]
        precision = (1 - all_nonzero / all) * 100
        print("测试精度:{}%".format(precision))

    torch.save(vgg.state_dict(), f"./saves/VGG-EP{i}.pth")
