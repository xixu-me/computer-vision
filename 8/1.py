# 将世界坐标系中的一个点投影到像素坐标系（世界坐标系->相机坐标系->图像坐标系->像素坐标系）

import numpy as np

intri = [[1111.0, 0.0, 400.0], [0.0, 1111.0, 400.0], [0.0, 0.0, 1.0]]

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


p1 = np.array([0, 0, 0])
screen_p1 = project_to_pixel(p1)
print(screen_p1)
