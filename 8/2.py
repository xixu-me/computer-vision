# 将世界坐标系中从相机中心出发的一条射线上的多个点投影到像素坐标系，验证这些点的投影点相同

import matplotlib.pyplot as plt
import numpy as np

intri = np.array([[1111.0, 0.0, 400.0], [0.0, 1111.0, 400.0], [0.0, 0.0, 1.0]])
extri = np.array(
    [
        [-9.9990e-01, 4.1922e-03, -1.3346e-02, -5.3798e-02],
        [-1.3989e-02, -2.9966e-01, 9.5394e-01, 3.8455e00],
        [-4.6566e-10, 9.5404e-01, 2.9969e-01, 1.2081e00],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def project_to_pixel(world_point):
    R = extri[:3, :3]
    T = extri[:3, -1]
    cam_p1 = np.dot(world_point - T, R)
    pixel_point = np.array(
        [
            -cam_p1[0] * intri[0][0] / cam_p1[2] + intri[0][2],
            cam_p1[1] * intri[1][1] / cam_p1[2] + intri[1][2],
        ]
    )
    return pixel_point


camera_origin = extri[:3, -1]
direction = extri[:3, :3].dot(np.array([1, 1, 10]))
direction /= np.linalg.norm(direction)
scale_factors = np.linspace(1, 10, 10)
for scale in scale_factors:
    sampled_point = camera_origin + scale * direction
    pixel_point = project_to_pixel(sampled_point)
    print(f"Sampled Point: {sampled_point}, Pixel Coordinate: {pixel_point}")
sampled_points = np.array(
    [camera_origin + scale * direction for scale in scale_factors]
)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    camera_origin[0],
    camera_origin[1],
    camera_origin[2],
    color="red",
    label="Camera Origin",
)
ax.text(camera_origin[0], camera_origin[1], camera_origin[2], "Camera", color="red")
ax.plot(
    sampled_points[:, 0],
    sampled_points[:, 1],
    sampled_points[:, 2],
    "bo-",
    label="Sample Points on Ray",
)
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
