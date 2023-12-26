import torch
import numpy as np
import torch.nn as nn

from functools import reduce

import torch.nn.functional as F

def pixel_accuracy(input, target):
    """像素精度"""
    N,H,W = target.size()
    softmaxed = torch.softmax(input, dim=1)
    return torch.sum(torch.argmax(softmaxed, dim=1) == target) / (N * H * W)

def pixel_accuracy_mean(input, target):
    """平均像素精度"""
    N,H,W = target.size()
    inputs = list(map(lambda i: torch.softmax(i, dim=1), input))
    sumed = list(map(lambda i: torch.sum(torch.argmax(i, dim=1) == target) / (N * H * W), inputs))
    summ = reduce(lambda ta, tb: torch.add(ta, tb), sumed)
    return summ / len(inputs)

def onehot(input, classes):
    """独热码转换"""
    ls = list(input.unsqueeze(1).shape)
    ls[1] = classes
    rst = torch.zeros(tuple(ls))
    rst = rst.scatter_(1, input.cpu(), 1)
    return rst

def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)

def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """计算批次图像中单个类别的DICE系数"""
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size

def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """计算批次图像中所有类别的平均DICE系数"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]

def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    """DICE损失"""
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

class ConfusionMatrix(object):
    """混淆矩阵"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k] + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return f"全局正确率：{(acc_global.item() * 100):.2f} 平均行正确率：{['{:.2f}'.format(i) for i in (acc * 100).tolist()]}\
            交并比：{['{:.2f}'.format(i) for i in (iu * 100).tolist()]} 平均交并比：{(iu.mean().item() * 100):.2f}"


class DiceCoefficient(object):
    """DICE系数"""
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # compute the Dice score, ignoring background
        pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index)
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)

