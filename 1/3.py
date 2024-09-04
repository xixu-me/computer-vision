import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

im = np.array(Image.open("images/house.jpg").convert("L"))
im2 = 255 - im
print("对图像进行反向处理:\n", int(im.min()), int(im.max()))
im3 = (100.0 / 255) * im + 100
print("将图像像素值变换到100...200区间:\n", int(im3.min()), int(im3.max()))
im4 = 255.0 * (im / 255.0) ** 2
print("对像素值求平方后得到的图像:\n", int(im4.min()), int(im4.max()))

plt.figure()
plt.gray()
plt.subplot(131)
plt.title("f(x) = 255 - x")
plt.imshow(im2)
plt.axis("off")
plt.subplot(132)
plt.title("f(x) = 100 / 255 * x + 100")
plt.imshow(im3)
plt.axis("off")
plt.subplot(133)
plt.title("f(x) = 255 * (x / 255) ^ 2")
plt.imshow(im4)
plt.axis("off")
plt.show()
