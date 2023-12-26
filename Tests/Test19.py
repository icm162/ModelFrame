import numpy as np
import cv2 as cv
import tqdm
import os

import tkinter as tk
from tkinter import filedialog

tk_root = tk.Tk()
tk_root.withdraw()

path = filedialog.askdirectory(
    initialdir="./",
    title="选择数据文件夹"
)

if path == "": exit()

bar = tqdm.tqdm(os.listdir(path))

for file_path in bar:

    if(not file_path.split(".")[-1].endswith("lbl")): continue
    abs_path = os.path.join(path, file_path)
    image = cv.imread(abs_path, cv.IMREAD_GRAYSCALE)
    image = np.uint8(np.where(image > 220, 255, 0))
    os.remove(abs_path)
    abs_path = abs_path.replace(".jpg", ".png")
    cv.imwrite(abs_path, image)


