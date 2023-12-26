import cv2 as cv
import os

img_path = r"C:\Users\icm162\Desktop\oed"

if __name__ == "__main__":

    images = [os.path.join(img_path, f) for f in os.listdir(img_path)]
    for file in images:
        image = cv.imread(file, cv.IMREAD_COLOR)
        

