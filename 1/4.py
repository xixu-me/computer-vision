import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
from scipy.ndimage import filters

plt.rcParams["font.sans-serif"] = ["SimSun"]
im = np.array(Image.open("images/house.jpg").convert("L"))

plt.figure()
plt.gray()
plt.subplot(141)
plt.title("原图")
plt.axis("off")
plt.imshow(im, cmap="gray")
for bi, blur in enumerate([2, 4, 8]):
    im2 = np.zeros(im.shape)
    im2 = filters.gaussian_filter(im, blur)  # type: ignore
    im2 = np.uint8(im2)
    imNum = str(blur)
    plt.subplot(1, 4, 2 + bi)
    plt.axis("off")
    plt.title("标准差为 " + imNum)
    plt.imshow(im2)
plt.show()
