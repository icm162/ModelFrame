import os
import time
import torch
import platform
from torch import nn
from tqdm import tqdm
import torchvision as tv

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 引入数据集类
# from UNetSEG.Datasets.GlaSDataset import GlaSDataset
# from UNetSEG.Datasets.GlaSTransDataset import GlaSTransDataset
# from UNetSEG.Datasets.LiverDataset import LiverDataset
# from UNetSEG.Datasets.CRAGCDataset import CRAGCDataset
# from UNetSEG.Datasets.ChessBoardDataset import ChessBoardDataset
from UNetSEG.Datasets.PDL1Dataset import PDL1Dataset

# 引入模型类
from UNetSEG.Models.UNet import UNet
# from UNetSEG.Models.FCCB import FCCB
# from UNetSEG.Models.UNetPP import UNetPlusPlus
# from UNetSEG.Models.UNetPPO import UnetPlusPlus as UNetPlusPlusOri
# from UNetSEG.Models.UNetPPCTR import UNetPlusPlusCTR
# from UNetSEG.Models.DV.vgg_unet import VGG16UNet
# from UNetSEG.Models.UNetPPCTRECA import UNetPlusPlusCTRECA
# from UNetSEG.Models.UPPCES import UNetPPCTRECASA
# from UNetSEG.Models.UPPATCES import UPPATCES

from UNetSEG.Models.TransAttUNet.TransAttUnet import UNet_Attention_Transformer_Multiscale
from UNetSEG.Models.TransUNet.vit_seg_modeling import VisionTransformer, CONFIGS

# 引入工具包
from UNetSEG.Utils.UtilsStandards import dice_loss, build_target
from UNetSEG.Utils.UtilsWindowsMessage import send_windows_message
from UNetSEG.Utils.UtilsLRScheme import deeplab_scheme_builder
from UNetSEG.Utils.UtilsPrint import tprint, cprint
import UNetSEG.Utils.UtilsCheckPoint as cp
import UNetSEG.Utils.UtilsProcess as process

"""训练 - 通过指定数据集训练指定模型"""

"""
    数据路径
    数据集类
    模型类
    模型序号
    读取最佳
    多端损失
    批次大小
    学习率
    训练轮数

    分割类别数
    模型初始化参数
"""
path_datasets = r"F:\DAT\PDL1"
dataset_class = PDL1Dataset
model_class = UNet
model_id = "pdl1-1"
read_besten = False
multi_out = False
batch_size = 1
learning_rate = 0.01
epochs = 100
contour_focus = False
loss_weight = torch.as_tensor([1.0, 2.0, 1.0])
contour_weight = torch.as_tensor([3.0, 2.0, 1.0])

num_classes = 3
model = model_class(3, num_classes, features=[16, 32])
# model = model_class(CONFIGS.get("R50-ViT-B_16"), 512)

# 模型定名_数据定名
model_name = f"{model_class.model_name}-{dataset_class.data_name}-{model_id}"
# 多输出时输出展示图数量
multi_num = len(model.features) if(hasattr(model, "features")) else 2
# 在未指定保存最终参数及读取最佳参数时读取的参数存档路径
name_load = "ruaruarua.pth"

# 数据集分支
path_dataparts = ["train", "val"]
# 存档路径
path_save = f"../Saves/{model_name}"
# 日志文件路径
path_log = "../Logs"

# 初始化目录
if(not os.path.exists(path_save)): os.makedirs(path_save)
if(not os.path.exists(path_log)): os.makedirs(path_log)
model_log_path = os.path.join(path_log, model_name)
if(not os.path.exists(model_log_path)): os.makedirs(model_log_path)
for file in os.listdir(path_log):
    sig = os.path.join(path_log, file)
    if(os.path.isfile(sig)): os.remove(sig)

# 验证周期
epochs_per_validate = 1
# 显示周期
batchs_per_display = 20
# 存档周期
epochs_per_save = 1
# 仅保留最后参数存档
save_last = True

# CUDA 可用标志
CUDA_FLAG = torch.cuda.is_available()

# 初始化白板日志器
writer = SummaryWriter(path_log)

# 损失函数
loss_weight = loss_weight.cuda() if(CUDA_FLAG) else loss_weight
contour_weight = contour_weight.cuda() if(CUDA_FLAG) else contour_weight
CE_Loss = lambda x, gt, w=loss_weight, wt=1: nn.functional.cross_entropy(x, gt, weight=w, ignore_index=255) * wt
DICE_Loss = dice_loss
Loss = lambda o, t, c, cf=contour_focus: CE_Loss(o, t) + (CE_Loss(o, c, contour_weight, 0.3) if(cf) else 0) \
     + DICE_Loss(o, build_target(t, num_classes, 255), multiclass=True)

# 优化器
params_with_grad = [param for param in model.parameters() if(param.requires_grad)]
optimizer = torch.optim.SGD(params_with_grad, momentum=0.9, lr=learning_rate, weight_decay=1e-4)

if(__name__ == "__main__"):

    cprint(f"\nCUDA {'' if(CUDA_FLAG) else '不'}可用\n", "info" if(CUDA_FLAG) else "error")
    model = model.cuda() if(CUDA_FLAG) else model

    # 加载数据集
    tr_data = dataset_class(path_datasets, path_dataparts[0])
    ts_data = dataset_class(path_datasets, path_dataparts[1])
    tr_len,ts_len = len(tr_data), len(ts_data)
    cprint(f"训练集长度为 {tr_len}  测试集长度为 {ts_len}\n", "healthy")

    # 数据加载器
    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    ts_loader = DataLoader(ts_data, batch_size=1)

    # 学习率规划器
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=deeplab_scheme_builder(len(tr_loader), epochs, warmup=True)) 

    # 读档
    from_ep = 0
    if(read_besten):
        # 读取最佳
        save_list = os.listdir(path_save)
        if(len(save_list) != 0):
            besten_save = list(filter(lambda f: f.count("-BESTEN") == 1, os.listdir(path_save)))
            assert len(besten_save) == 1, "最佳参数存档缺失或多余"
            model, optimizer, lr_scheduler, epstr = cp.load(model, optimizer, lr_scheduler, path_save, besten_save[0])
            from_ep = int(epstr)
            cprint(f"已读取 {from_ep} 轮最佳训练参数", "healthy")
    elif(save_last):
        # 读取最终
        save_list = os.listdir(path_save)
        if(len(save_list) != 0):
            final_saves = list(filter(lambda f: f.count("-FINAL") == 1, os.listdir(path_save)))
            assert len(final_saves) == 1, "最终参数存档缺失或多余"
            model, optimizer, lr_scheduler, epstr = cp.load(model, optimizer, lr_scheduler, path_save, final_saves[0])
            from_ep = int(epstr)
            cprint(f"已读取 {from_ep} 轮最终训练参数", "warn")
    else:
        # 读取指定
        model, optimizer, lr_scheduler, epstr = cp.load(model, optimizer, lr_scheduler, path_save, name_load)
        from_ep = int(epstr)
        cprint(f"已读取 {from_ep} 轮普通训练参数", "unwill")

    print("\n")

    # 迭代轮次
    for epoch in range(epochs):

        # 轮标
        tprint(f"第 {from_ep + epoch + 1} 轮训练   本次训练第 {epoch + 1} / {epochs} 轮", "", "")

        # 轮初始化
        start_time = time.time()
        pas, mious, losss = [], [], []
        bar = tqdm(tr_loader)

        # 轮训练
        model.train()
        for i, datum in enumerate(bar):

            if(contour_focus): imgs, targets, contours = datum
            else:
                imgs, targets = datum
                contours = targets

            if(CUDA_FLAG): imgs, targets, contours = imgs.cuda(), targets.cuda(), contours.cuda()

            # 训练单批次
            model, optimizer, loss, outs, matrix = process.train_batch(model, optimizer, lr_scheduler, multi_out, num_classes, imgs, targets, contours, True, True)

            # 更新进度条显示
            bar.set_postfix(耗时=f"{(time.time() - start_time):.3f}s")

            # 更新并记录指标
            pa = process.pixel_accuracy_mean(outs, targets) if(multi_out) else process.pixel_accuracy(outs, targets)
            miou = matrix.compute()[2].mean().item() * 100
            blr = optimizer.param_groups[0]["lr"]
            pas.append(pa)
            mious.append(miou)
            losss.append(loss)
            writer.add_scalar("训练损失", loss, epoch * len(tr_loader) + i + 1)
            writer.add_scalar("像素精度", pa, epoch * len(tr_loader) + i + 1)
            writer.add_scalar("平均交并比", miou, epoch * len(tr_loader) + i + 1)
            writer.add_scalar("学习率", blr, epoch * len(tr_loader) + i + 1)

            mean = lambda ls: sum(ls) / len(ls)

            # 打印累积指标
            if((i + 1) % batchs_per_display == 0 or i == len(tr_loader) - 1):
                print(f"\n第 {i + 1} 批次训练 - 损失为 {mean(losss):.3f} 像素精度为 {mean(pas):.2f} 平均交并比 {mean(mious):.2f} 学习率 {blr:.6f} 总用时 {(time.time() - start_time):.3f}s")
                pas, mious, losss = [], [], []

        # 存档
        if((epoch + 1) % epochs_per_save == 0): cp.save(model, optimizer, lr_scheduler, model_name, from_ep + epoch + 1, path_save, save_last)

        # 判断是否验证
        if((epoch + 1) % epochs_per_validate != 0): continue
        print("\n")

        # 完成单轮验证
        model.eval()
        mPA, mIOU, smIOU, dice = process.validate(ts_loader, model, from_ep + epoch + 1, multi_out, num_classes, CUDA_FLAG, True)

        # 储存最佳验证参数
        cond = lambda mt, cp: mt >= float(cp)
        best_list = list(filter(lambda f: f.count("-BESTEN") == 1, os.listdir(path_save)))
        assert len(best_list) == 1 or len(best_list) == 0, "不允许出现多个最佳参数存档"
        if(len(best_list) == 0): cp.save(model, optimizer, lr_scheduler, model_name, from_ep + epoch + 1, path_save, False, (dice, smIOU))
        elif(cond(dice, best_list[0].split("-")[-2])): 
            os.remove(os.path.join(path_save, best_list[0]))
            cp.save(model, optimizer, lr_scheduler, model_name, from_ep + epoch + 1, path_save, False, (dice, smIOU))
            
    # 训练结束存档
    if(save_last): cp.save(model, optimizer, lr_scheduler, model_name, from_ep + epoch + 1, path_save, True)
    cprint("\n最终状态存档完成\n", "info")
    
    # 获取最佳存档信息
    bests = list(filter(lambda f: f.count("-BESTEN") == 1, os.listdir(path_save)))
    assert len(bests) == 1, "最佳参数存档数量有误"

    # 发送系统提示消息
    if(platform.system() == "Windows"):
        send_windows_message(f"{model_name}训练完成", f"完成了{from_ep + epochs}轮训练\n最大平均交并比达到 {float(bests[0].split('-')[-1].replace('.pth', ''))}")

    # 挂起白板日志
    writer.close()
    os.system(f"tensorboard --logdir={path_log}")





