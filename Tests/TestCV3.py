import cv2 as cv
import numpy as np

img = np.array([[127]*200]*100+[[0]*200]*100, dtype=np.uint8)

kernel_size = (7, 7)
iterations = 16
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

print(img.shape)
imge = cv.erode(img, kernel, iterations=iterations)
cv.imshow("img", img)
cv.imshow("imge", imge)
cv.waitKey(0)
cv.destroyAllWindows()