import torch
import cv2 as cv
import numpy as np

path = r"F:\DAT\GlaST\train\013_mask.bmp"
kernel_size = (7, 7)
iterations = 10
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

def contour_focus(ori_img:np.ndarray):

    img = ori_img.copy()
    imge = cv.erode(img, kernel, iterations=iterations)
    imgd = cv.dilate(img, kernel, iterations=iterations)

    img_delta = imgd - imge
    img[img == 255] = 127
    img = np.where(img_delta==0, 255, cv.bitwise_and(img, img_delta))
    img = np.where(img!=255, img // 127, img)

    return img

def contour_focus_path(path:str):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    return contour_focus(img)

if(__name__ == "__main__"):
    cv.imshow("cf", contour_focus_path(path))
    cv.waitKey(0)
    cv.destroyAllWindows()




