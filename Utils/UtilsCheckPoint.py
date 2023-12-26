import torch
import re
import os

from collections import Iterable

from UNetSEG.Utils.UtilsPrint import cprint

def save(model, optimizer, lr_scheduler, model_name, epoch, path, savelast=True, reach=None):
    plain_save = (not savelast) and reach == None
    save_contents = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch
    }
    if(savelast):
        for file in list(filter(lambda f: f.count("-FINAL") == 1, os.listdir(path))): os.remove(os.path.join(path, file))
    if(reach != None):
        stands = "-".join(list(map(lambda r: str(round(r, 3)), reach))) if(isinstance(reach, Iterable)) else str(round(reach, 3))
    torch.save(save_contents, os.path.join(path, \
        f"{model_name}-EP{epoch}{'-FINAL' if(savelast) else '' if(plain_save) else '-BESTEN-' + stands}.pth"))
    cprint("\n\n自动存档完成\n\n", "info")

def load(model, optimizer, lr_scheduler, path, filename=None, location="cpu"):
    filelist = os.listdir(path)
    if(filename is None):
        assert len(filelist) == 1, "读档参数不可省略"
        filename = filelist[0]
    loaded_contents = torch.load(os.path.join(path, filename), map_location=location)
    model.load_state_dict(loaded_contents["model"])
    optimizer.load_state_dict(loaded_contents["optimizer"])
    lr_scheduler.load_state_dict(loaded_contents["lr_scheduler"])
    valep = loaded_contents["epoch"]
    epstr = list(filter(lambda n: re.match("EP\d+", n) != None, filename.split("-")))
    assert len(epstr) == 1 or valep != None, "读取存档缺少明确轮次记录"
    epstr = epstr[0][2:]
    if(valep != None and str(valep) != epstr): 
        cprint("警告：参数存档称名轮次与记录轮次不一致\n", "warn")
        epstr = str(valep)
    return model, optimizer, lr_scheduler, epstr

def load_model(model, path, filename, location="cpu"):
    abs_path = os.path.join(path, filename)
    assert os.path.isfile(abs_path), "参数存档路径有误"
    loaded_contents = torch.load(abs_path, map_location=location)
    model.load_state_dict(loaded_contents["model"])
    valep = loaded_contents["epoch"]
    return model, valep
