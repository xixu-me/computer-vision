import cv2 as cv
import numpy as np
import pywt
from matplotlib import pyplot as plt

img = cv.imread("images/house.jpg")
img = cv.resize(img, (512, 512))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)

plt.figure("二维小波一级变换")
coeffs = pywt.dwt2(img, "haar")
cA, (cH, cV, cD) = coeffs

AH = np.concatenate([cA, cH], axis=1)
VD = np.concatenate([cV, cD], axis=1)
img1 = np.concatenate([AH, VD], axis=0)

plt.imshow(img1, cmap="gray")
plt.title("result")
plt.axis("off")
plt.show()
