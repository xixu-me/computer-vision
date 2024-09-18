import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pywt

img = cv.imread("images/house.jpg")
img = cv.resize(img, (512, 512))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)

plt.figure("二维图像多级分解")
coeffs = pywt.wavedec2(img, "haar", level=2)
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

AH2 = np.concatenate([cA2, cH2 + 510], axis=1)
VD2 = np.concatenate([cV2 + 510, cD2 + 510], axis=1)
cA1 = np.concatenate([AH2, VD2], axis=0)

AH = np.concatenate([cA1, (cH1 + 255) * 2], axis=1)
VD = np.concatenate([(cV1 + 255) * 2, (cD1 + 255) * 2], axis=1)
img = np.concatenate([AH, VD], axis=0)

plt.imshow(img, cmap="gray")
plt.title("2D WT")
plt.axis("off")
plt.show()
