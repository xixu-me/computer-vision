import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rows, cols, channels = image.shape
    image_reshaped = image.reshape(rows * cols, channels)
    return image, image_reshaped


def initialize_centroids(data, k):
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]
    return centroids


def assign_clusters(data, centroids):

    labels = []
    for i in range(len(data)):
        min_distance = float("inf")
        closest_centroid = -1
        for j in range(len(centroids)):
            distance = np.sum((data[i] - centroids[j]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = j
        labels.append(closest_centroid)
    return np.array(labels)


def update_centroids(data, labels, k):
    centroids = []
    for i in range(k):
        cluster_points = []
        for j in range(len(data)):
            if labels[j] == i:
                cluster_points.append(data[j])
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = data[np.random.choice(len(data))]
        centroids.append(new_centroid)
    return np.array(centroids)


def kmeans(data, k, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        old_centroids = centroids
        labels = assign_clusters(data, centroids)
        centroids = update_centroids(data, labels, k)
        total_movement = 0
        for i in range(len(centroids)):
            movement = 0
            for j in range(len(centroids[i])):
                movement += (old_centroids[i][j] - centroids[i][j]) ** 2
            movement = movement**0.5
            total_movement += movement
        if total_movement < tolerance:
            break
    return labels, centroids


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


image, image_data = read_image("images/small.png")
rows, cols, _ = image.shape
labels, centers = kmeans(image_data, 5)
display_segmented_image(image, labels, centers, rows, cols)
