import os
import cv2 as cv
from pyselflib import header_image as hi

template = r"F:\PTC\UNetSEG\Predict\GlaS\DP-U-Net++"
path = r"F:\PTC\UNetSEG\Predict\GlaS\TransAttUNet"
sizes = []

for root, dirs, files in os.walk(template):
    for file in files:
        sizes.append(hi.get_png_size(os.path.join(root, file)))

for root, dirs, files in os.walk(path):
    for i, file in enumerate(files):
        file_path = os.path.join(root, file)
        cv.imwrite(file_path, cv.resize(cv.imread(file_path, cv.IMREAD_GRAYSCALE), sizes[i], interpolation=cv.INTER_LINEAR))

