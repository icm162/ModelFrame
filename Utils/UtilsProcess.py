import os
import time
import torch
import torchvision as tv
import torchvision.transforms.functional as F

from UNetSEG.Utils.UtilsStandards import ConfusionMatrix, DiceCoefficient, pixel_accuracy, pixel_accuracy_mean
from UNetSEG.Opts.Train import Loss, model_log_path


def train_batch(model, optimizer, lr_scheduler, multi_out, num_classes, imgs, targets, contours, save_data=False, save_outs=True, debug=False):

    matrix = ConfusionMatrix(num_classes)

    if(debug): 
        print(f"\n{imgs.shape}\n{targets.shape}\n")
        print(f"\n{imgs.dtype}\n{targets.dtype}\n")

    if(save_data):
        for i in range(imgs.shape[0]):
            tv.utils.save_image(imgs[i, ...], f"../Segmented/dbg/img-{i + 1}.jpg")
            tv.utils.save_image(targets[i, ...].to(torch.float16), f"../Segmented/dbg/target-{i + 1}.jpg")

    if(multi_out):
        out_list = model(imgs)
        outs = list(map(lambda o: torch.sigmoid(o), out_list))
        losses = list(map(lambda o: Loss(o, targets, contours), outs))
        for out in outs: matrix.update(targets.flatten(), out.argmax(1).flatten())
    else:
        out = model(imgs)
        out = out["out"] if(isinstance(out, dict)) else out
        sout = torch.sigmoid(out)
        losses = Loss(sout, targets, contours)
        matrix.update(targets.flatten(), out.argmax(1).flatten())

    if(debug):
        print("输出形状:")
        print(torch.round(outs if(multi_out) else sout).shape)
        print("标签形状:")
        print(targets.shape)

    if(save_outs):
        if(multi_out):
            for i, oi in enumerate(outs): 
                for j in range(oi.shape[0]): tv.utils.save_image(oi[j, ...].argmax(0).to(torch.float16), f"../Segmented/dbg/out-{i + 1}-{j + 1}.jpg")
        else: 
            for j in range(out.shape[0]): tv.utils.save_image(out[j, ...].argmax(0).to(torch.float16), f"../Segmented/dbg/out-{j + 1}.jpg")
    
    optimizer.zero_grad()
    if(multi_out): 
        for loss in losses: loss.backward(retain_graph=True)
    else: losses.backward()
    optimizer.step()
    lr_scheduler.step()

    matrix.reduce_from_all_processes()

    loss = losses.item() if(not multi_out) else sum(list(map(lambda l: l.item(), losses))) / len(losses)

    return model, optimizer, loss, (outs if(multi_out) else sout), matrix


def validate(loader, model, epoch_num, multi_out=True, num_classes=2, CUDA_FLAG=True, save_img=True, resize=True, show_num=6):
    avg_acc, miou, num = 0, 0, 0
    start_time = time.time()
    ls_mious = []
    with torch.no_grad():
        sum_matrix = ConfusionMatrix(num_classes)
        dice_coef = DiceCoefficient(num_classes, 255)
        for i, datum in enumerate(loader):
            imgs, targets = datum[:2]
            if(CUDA_FLAG): imgs,targets = imgs.cuda(), targets.cuda()
            if(multi_out):
                out_list = model(imgs)
                outs, ious = list(map(lambda o: torch.sigmoid(o), out_list)), []
                for oo in outs:
                    if(resize and oo.shape[-2:] != targets.shape[-2:]): oo = F.resize(oo, targets.shape[-2:], F.InterpolationMode.NEAREST)
                    matrix = ConfusionMatrix(num_classes)
                    matrix.update(targets.flatten(), oo.argmax(1).flatten())
                    sum_matrix.update(targets.flatten(), oo.argmax(1).flatten())
                    dice_coef.update(oo, targets)
                    matrix.reduce_from_all_processes()
                    ious.append(matrix.compute()[2].mean().item() * 100)
                mious = max(ious) # reduce(lambda a, b: a + b, ious) / len(ious)
            else:
                out = model(imgs)
                rout = out["out"] if(isinstance(out, dict)) else out
                sout = torch.sigmoid(rout)
                if(resize and sout.shape[-2:] != targets.shape[-2:]): sout = F.resize(sout, targets.shape[-2:], F.InterpolationMode.NEAREST)
                matrix = ConfusionMatrix(num_classes)
                matrix.update(targets.flatten(), sout.argmax(1).flatten())
                sum_matrix.update(targets.flatten(), sout.argmax(1).flatten())
                dice_coef.update(sout, targets)
                matrix.reduce_from_all_processes()
                mious = matrix.compute()[2].mean().item() * 100

            miou, num = miou * num / (num + 1) + mious / (num + 1), num + 1
            ls_mious.append(str(round(mious, 2)))

            if((i + 1) % show_num == 0 or (i + 1) == len(loader)):
                print(f"验证 - {i + 2 - show_num}~{i + 1 if(i + 1 + show_num <= len(loader)) else len(loader)} / {len(loader)} 平均记录交并比 {'   '.join(ls_mious)}")
                ls_mious = []

            avg_acc += pixel_accuracy_mean(outs, targets) if(multi_out) else pixel_accuracy(sout, targets)
            if(save_img):
                outs_list = outs if(multi_out) else [sout]
                for j, ot in enumerate(outs_list): 
                    for k in range(ot.shape[0]): tv.utils.save_image(ot[k, ...].argmax(0).to(torch.float16), f"../Segmented/{i + 1}-{j + 1}-{k + 1}.jpg")
        
    sum_matrix.reduce_from_all_processes()
    dice_coef.reduce_from_all_processes()
    sum_miou = sum_matrix.compute()[2].mean().item() * 100
    dice_coefficient = dice_coef.value.item()

    mPA, mIOU = avg_acc / len(loader), miou
    
    with open(os.path.join(model_log_path, "sk.log"), "a", encoding="utf-8") as file:
        file.write(f"{str(epoch_num).zfill(8)} 轮次 - 平均像素准确率 {mPA:.4f} 平均记录交并比 {mIOU:.4f} 平均交并比 {sum_miou:.4f} DICE系数 {dice_coefficient:.4f}\n")

    print(f"\n\n验证 - 平均像素准确率为 {mPA:.2f} 平均记录交并比为 {mIOU:.2f} 平均交并比 {sum_miou:.2f} DICE系数 {dice_coefficient:.2f} 用时 {(time.time() - start_time):.3f}s")

    return mPA, mIOU, sum_miou, dice_coefficient

if(__name__ == "__main__"):
    """测试验证过程是否正确"""
    validate(None, None, 2, True, True)

