import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def read_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rows, cols, channels = image.shape
    image_reshaped = image.reshape(rows * cols, channels)
    return image, image_reshaped


def apply_kmeans(image_data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(image_data)
    centers = kmeans.cluster_centers_
    return labels, centers


def display_segmented_image(image, labels, centers, rows, cols):
    centers = np.uint8(centers)
    segmented_image = centers[labels]  # type: ignore
    segmented_image = segmented_image.reshape((rows, cols, 3))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")
    plt.show()


image, image_data = read_image("images/lena-color.png")
rows, cols, _ = image.shape
labels, centers = apply_kmeans(image_data, 5)
display_segmented_image(image, labels, centers, rows, cols)
