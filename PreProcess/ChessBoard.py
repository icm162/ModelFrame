import numpy as np
import cv2 as cv
import os

path = r"F:\DAT\ChessBoard"
subs = [r"train", r"val"]
feature = "_lbl"
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
iterations = 5

for sub in subs:
    sub_path = os.path.join(path, sub)
    in_folder = list(filter(lambda f: f.count(feature) != 0, os.listdir(sub_path)))
    for file in in_folder:
        path_file = os.path.join(sub_path, file)
        rd = cv.imread(path_file, cv.IMREAD_REDUCED_GRAYSCALE_8)
        cv.imshow("rd", rd)
        cv.waitKey(0)
        image = cv.imread(path_file, cv.IMREAD_COLOR)
        image = cv.dilate(image, kernel, iterations=iterations)
        cv.imwrite(path_file, image)
        


