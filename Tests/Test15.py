import cv2 as cv

"""反色"""

path = r"C:\Users\icm162\Desktop\contour.png"
to = r"C:\Users\icm162\Desktop\contour-r.png"

image = cv.imread(path, cv.IMREAD_GRAYSCALE)
reverse = 255 - image
cv.imwrite(to, reverse)


