import cv2 as cv
import numpy as np

path = r"F:\DAT\GlaST\train\010_mask.bmp"

kernel_size = (7, 7)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

img = cv.imread(path, cv.IMREAD_GRAYSCALE)

imge = cv.erode(img, kernel, iterations=5)
imgd = cv.dilate(img, kernel, iterations=5)

img = imgd - imge

img = cv.GaussianBlur(img, kernel_size, 0, sigmaY=0)

print(np.max(img), np.min(img))

cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()