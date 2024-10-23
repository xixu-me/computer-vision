# 使用sklearn.cluster库中的MeanShift类，对生成的二维模拟数据进行聚类。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.4)

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=300)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    s=300,
    c="red",
    marker="*",
    edgecolor="k",
)
plt.title("Mean Shift Clustering")
plt.show()
