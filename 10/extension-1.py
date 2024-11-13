# 利用OpenCV实现多张2D图像的3D重构。每张图片的内外方位元素从templeRing_par.txt文件中读取

import cv2
import numpy as np
import plotly.express as px
import plotly.io as pio

# 设置plotly在浏览器中显示
pio.renderers.default = "browser"


def read_camera_parameters(file_path):
    """读取相机参数文件"""
    cameras = []
    with open(file_path, "r") as f:
        n_cameras = int(f.readline())
        for _ in range(n_cameras):
            line = f.readline().split()
            camera = {
                "image": line[0],
                "K": np.array(
                    [
                        [float(line[1]), 0, float(line[3])],
                        [0, float(line[5]), float(line[6])],
                        [0, 0, 1],
                    ]
                ),
                "R": np.array(
                    [
                        [float(line[10]), float(line[11]), float(line[12])],
                        [float(line[13]), float(line[14]), float(line[15])],
                        [float(line[16]), float(line[17]), float(line[18])],
                    ]
                ),
                "t": np.array(
                    [[float(line[19])], [float(line[20])], [float(line[21])]]
                ),
            }
            cameras.append(camera)
    return cameras


def reconstruct_3d_points(img1, img2, K, R1, t1, R2, t2):
    """从两张图片重构3D点云"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和计算描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择前N个最佳匹配点
    N_BEST_MATCHES = 100
    matches = matches[:N_BEST_MATCHES]

    # 提取匹配点的坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算投影矩阵
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    # 使用RANSAC进行几何验证
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    # 只保留内点
    matches = [m for i, m in enumerate(matches) if mask[i][0]]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # 三角测量得到3D点
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_4D / points_4D[3]
    points_3D = points_3D[:3].T

    # 可选：显示匹配结果
    img_matches = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(1)  # 显示1毫秒

    return points_3D


def main():
    # 读取相机参数
    cameras = read_camera_parameters(r"10\templeRing\templeR_par.txt")

    # 初始化空数组存储所有3D点
    all_points_3D = []

    # 从连续的图像对重构3D点
    for i in range(len(cameras) - 1):
        print(f"Processing image pair {i+1}/{len(cameras)-1}")

        # 读取图像
        img1 = cv2.imread(f"10/templeRing/{cameras[i]['image']}")
        img2 = cv2.imread(f"10/templeRing/{cameras[i+1]['image']}")

        if img1 is None or img2 is None:
            print(f"Failed to load images {i} and {i+1}")
            continue

        points_3D = reconstruct_3d_points(
            img1,
            img2,
            cameras[i]["K"],
            cameras[i]["R"],
            cameras[i]["t"],
            cameras[i + 1]["R"],
            cameras[i + 1]["t"],
        )

        # 过滤异常点
        distances = np.sqrt(np.sum(points_3D**2, axis=1))
        mask = distances < np.percentile(distances, 95)  # 移除最远的5%的点
        points_3D = points_3D[mask]

        all_points_3D.append(points_3D)

    # 关闭所有窗口
    cv2.destroyAllWindows()

    # 组合所有点
    all_points_3D = np.vstack(all_points_3D)

    # 使用plotly进行3D可视化
    fig = px.scatter_3d(
        x=all_points_3D[:, 0],
        y=all_points_3D[:, 1],
        z=all_points_3D[:, 2],
        title="3D Reconstruction from Multiple Views",
    )

    # 调整点的大小和视角
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(scene=dict(aspectmode="data"))  # 保持真实比例

    # 显示结果
    fig.show()

    # 保存结果
    fig.write_html("10/templeR_multi.html")


if __name__ == "__main__":
    main()
