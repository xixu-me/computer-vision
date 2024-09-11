import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
from skimage.restoration import denoise_tv_chambolle

# import PCV.tools as rof

plt.rcParams["font.sans-serif"] = ["SimSun"]
im = np.zeros((500, 500))
im[100:400, 100:400] = 128
im[200:300, 200:300] = 255
im = im + 30 * np.random.standard_normal((500, 500))
U = denoise_tv_chambolle(im, weight=0.1)
# U, T = rof.denoise(im, im)
G = filters.gaussian_filter(im, 10)  # type: ignore

plt.figure()
plt.gray()
plt.subplot(131)
plt.title("原噪声图像")
plt.axis("off")
plt.imshow(im)
plt.subplot(132)
plt.title("高斯模糊后的图像")
plt.axis("off")
plt.imshow(G)
plt.subplot(133)
plt.title("ROF 降噪后的图像")
plt.axis("off")
plt.imshow(U)
plt.tight_layout()
plt.show()
