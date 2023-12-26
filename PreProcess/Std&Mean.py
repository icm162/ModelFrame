import numpy as np
import tqdm
import cv2
import os

# img_h, img_w = 32, 32
# img_h, img_w = 512,512  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = r'F:\DAT\PH2\train\image'
imgs_path_list = [i for i in os.listdir(imgs_path) if i.count("_mask")==0]

bar = tqdm.tqdm(imgs_path_list)

for item in bar:
    img = cv2.imread(os.path.join(imgs_path, item))
    # img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# 但是CV到CV不变
# means.reverse()
# stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
