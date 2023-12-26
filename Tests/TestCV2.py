import cv2 as cv
import numpy as np

path = r"F:\DAT\GlaST\train\013_mask.bmp"

kernel_size = (7, 7)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

img = cv.imread(path, cv.IMREAD_GRAYSCALE)

imge = cv.erode(img, kernel, iterations=5)
imgd = cv.dilate(img, kernel, iterations=5)

img_delta = imgd - imge
img[img == 255] = 127
img = np.where(img_delta==0, 255, cv.bitwise_and(img, img_delta))
img = np.where(img!=255, img // 127, img)

cv.imshow("img", img)
cv.imshow("img_delta", img_delta)

cv.waitKey(0)
cv.destroyAllWindows()



