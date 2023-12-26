import numpy as np
import cv2 as cv


path = r"C:\Users\icm162\Desktop\001c62abd11fa4b57bf7a6c603a11bb9.png"


image = cv.imread(path, cv.IMREAD_REDUCED_COLOR_8)
size = list(map(lambda i: i // 4, image.shape[:2]))
image = cv.resize(image, size)
cv.imshow("Image", image)
cv.waitKey(0)
cv.destroyAllWindows()


