import numpy as np
import cv2 as cv

a = np.array([[255, 255, 0, 0]], dtype=np.uint8)
b = np.array([[255, 0, 255, 0]], dtype=np.uint8)

print(np.uint8(np.where(np.greater(a, b), 200, 100)))

