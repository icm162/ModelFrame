import numpy as np
import cv2 as cv
import os

path = r"F:\DAT\CRAGO"
to_path = r"F:\DAT\CRAGR"

l1 = ["train", "val"]
l2 = ["image", "label"]

to_size = (512, 512)

for ds in l1:
    for ic in l2:
        folder = os.path.join(path, ds, ic)
        to_folder = os.path.join(to_path, ds, ic)
        for img in os.listdir(folder):
            image = cv.imread(os.path.join(folder, img))
            if(ic == "label"): image = np.uint8(np.where(image != 0, 255, 0))
            image = cv.resize(image, to_size, interpolation=(cv.INTER_LINEAR if(ic == "image") else cv.INTER_NEAREST))
            cv.imwrite(os.path.join(to_folder, img), image)




