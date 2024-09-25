import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread("images/lena.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

block_size = 2
sobel_size = 3
k = 0.04

corners_img = cv.cornerHarris(gray, block_size, sobel_size, k)  # type: ignore

indices = np.argsort(corners_img.flatten())[::-1][:50]
coords = np.column_stack(np.unravel_index(indices, corners_img.shape))
for coord in coords:
    cv.circle(image, (coord[1], coord[0]), 3, (255, 0, 0), 1)

plt.imshow(image)
plt.axis("off")
plt.tight_layout()
plt.show()
