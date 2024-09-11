import cv2 as cv
from matplotlib import pyplot as plt


def build_gaussian_pyramid(image, levels):
    pyramid = [image]
    temp = image
    for i in range(levels):
        blurred = cv.GaussianBlur(temp, (5, 5), 0)
        down_sampled = cv.resize(
            blurred,
            (temp.shape[1] // 2, temp.shape[0] // 2),
            interpolation=cv.INTER_LINEAR,
        )
        pyramid.append(down_sampled)
        temp = down_sampled
    return pyramid


image = cv.imread("images/lena.jpg", cv.IMREAD_GRAYSCALE)
pyramid = build_gaussian_pyramid(image, 3)

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.figure(figsize=(12, 6))
for i in range(len(pyramid)):
    plt.subplot(1, len(pyramid), i + 1)
    plt.imshow(pyramid[i], cmap="gray")
    plt.title("Level " + str(i))
    plt.axis("off")
plt.tight_layout()
plt.show()
