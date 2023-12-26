import os
import tqdm
import cv2 as cv
import numpy as np

path = r"F:\DAT\CRAG-V2\valid"
to_path = r"F:\DAT\CRAGC\val"

froms = ("Images", "Annotation")
tos = ("image", "label")

to_size = 512
threshold = 232

filterr = lambda f: f.endswith(".png")

bar = tqdm.tqdm(list(filter(filterr, os.listdir(os.path.join(path, froms[0])))))

for p in bar:
    img = cv.imread(os.path.join(path, froms[0], p), cv.IMREAD_COLOR)
    lab = cv.imread(os.path.join(path, froms[1], p), cv.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    h_split = (h - to_size) // to_size if((h - to_size) % to_size == 0) else (h - to_size) // to_size + 1
    w_split = (w - to_size) // to_size if((w - to_size) % to_size == 0) else (w - to_size) // to_size + 1
    h_ls = [round((h - to_size) * hh / h_split) for hh in range(h_split + 1)]
    w_ls = [round((w - to_size) * ww / w_split) for ww in range(w_split + 1)]
    for i, hh in enumerate(h_ls):
        for j, ww in enumerate(w_ls):
            sub_img = img[hh:hh+to_size, ww:ww+to_size, ...]
            if(np.mean(sub_img) > threshold): continue
            sub_lab = lab[hh:hh+to_size, ww:ww+to_size, ...]
            sub_lab = np.where(sub_lab > 0, 255, 0)
            save_name = p.replace(".png", f"_{i*(w_split+1) + j + 1}.png")
            cv.imwrite(os.path.join(to_path, tos[0], save_name), sub_img)
            cv.imwrite(os.path.join(to_path, tos[1], save_name), sub_lab)
