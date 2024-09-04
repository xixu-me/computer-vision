from matplotlib import pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["SimSun"]

pil_im = Image.open("images/house.jpg")
pil_im1 = Image.open("images/house.jpg").convert("L")

plt.figure()
plt.subplot(121)
plt.title("原图")
plt.axis("off")
plt.imshow(pil_im)
plt.subplot(122)
plt.title("灰度图")
plt.axis("off")
plt.imshow(pil_im1, cmap="gray")
plt.show()
