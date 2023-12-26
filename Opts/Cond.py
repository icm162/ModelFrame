from PIL import Image
import numpy as np
import cv2 as cv
import tqdm
import os
import re

label_path = r"F:\DAT\PH2\val\label"
dataset_name = "PH2"

def cond(predict_path, label_path, save_path, rgb_01=(255, 0, 0), rgb_10=(0, 0, 255)):

    lbl_ls = list(map(lambda f: os.path.join(label_path, f), os.listdir(label_path)))
    # sorted(os.listdir(label_path), key=lambda k: int(re.findall("\d+", k)[0]))))
    pred_ls = list(map(lambda f: os.path.join(predict_path, f), sorted(os.listdir(predict_path), key=lambda f: int(f.split("-")[1].split(".")[0]))))

    lbl_tqdm = tqdm.tqdm(lbl_ls)

    for i, file_path in enumerate(lbl_tqdm):

        # 防止 jpg 读取问题 加入二值化处理
        _, predict = cv.threshold(cv.imread(file_path, cv.IMREAD_GRAYSCALE), 127, 255, cv.THRESH_BINARY)
        _, label = cv.threshold(cv.imread(pred_ls[i], cv.IMREAD_GRAYSCALE), 127, 255, cv.THRESH_BINARY)

        conded = np.uint8(np.where(predict == label, predict, np.where(predict > label, 200, 100)))

        r = np.uint8(np.where(conded == 100, rgb_01[0], np.where(conded == 200, rgb_10[0], conded)))
        g = np.uint8(np.where(conded == 100, rgb_01[1], np.where(conded == 200, rgb_10[1], conded)))
        b = np.uint8(np.where(conded == 100, rgb_01[2], np.where(conded == 200, rgb_10[2], conded)))
        
        cv.imwrite(os.path.join(save_path, f"{i + 1}.png"), cv.merge([b, g, r]))

if __name__ == "__main__":
    predict_path = r"../Predict"
    for model_name in os.listdir(os.path.join(predict_path, dataset_name)):
        predict_path = r"../Predict"
        save_path = os.path.join(r"../Cond", dataset_name, model_name)
        if(not os.path.exists(save_path)): os.mkdir(save_path)
        else: 
            for file in os.listdir(save_path): os.remove(os.path.join(save_path, file))
        predict_path = os.path.join(predict_path, dataset_name, model_name)
        cond(predict_path, label_path, save_path, (255, 192, 64), (128, 128, 255))
    # os.startfile(save_path)