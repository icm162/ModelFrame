import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def pad_if_smaller(img, size, fill=0):

    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, *args):
        inn = [image, target] + list(args)
        for t in self.transforms: inn = t(*inn)
        return inn

class RandomResize(object):

    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, *args):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        more = [F.resize(p, size, interpolation=T.InterpolationMode.NEAREST) for p in args]
        return [image, target] + more

class ForceResize(object):

    def __init__(self, h_size, w_size=None):
        self.h_size = h_size
        self.w_size = h_size if(w_size is None) else w_size 

    def __call__(self, image, target, *args):
        image = F.resize(image, (self.h_size, self.w_size))
        # target = F.resize(target, (self.h_size, self.w_size), interpolation=T.InterpolationMode.NEAREST)
        more = [F.resize(p, (self.h_size, self.w_size), interpolation=T.InterpolationMode.NEAREST) for p in args]
        return [image, target] + more

class RandomHorizontalFlip(object):
    
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            more = [F.hflip(p) for p in args]
            return [image, target] + more
        return [image, target] + list(args)

class RandomVerticalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, *args):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            more = [F.vflip(p) for p in args]
            return [image, target] + more
        return [image, target] + list(args)

class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, *args):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        more = [pad_if_smaller(p, self.size, fill=255) for p in args]
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        more = [F.crop(p, *crop_params) for p in more]
        return [image, target] + more

class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, *args):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        more = [F.center_crop(p, self.size) for p in args]
        return [image, target] + more

class ToTensor(object):

    def __call__(self, image, target, *args):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        more = [torch.as_tensor(np.array(p), dtype=torch.int64) for p in args]
        return [image, target] + more

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, *args):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return [image, target] + list(args)

class ToPILImage(object):
    def __call__(self, image, target, *args):
        image = F.to_pil_image(image)
        target = F.to_pil_image(target)
        more = [F.to_pil_image(p) for p in args]
        return [image, target] + more
