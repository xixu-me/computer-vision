import cv2
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"


def read_camera_parameters(file_path):
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
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    N_BEST_MATCHES = 100
    matches = matches[:N_BEST_MATCHES]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    matches = [m for i, m in enumerate(matches) if mask[i][0]]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_4D / points_4D[3]
    points_3D = points_3D[:3].T
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
    cv2.waitKey(1)
    return points_3D


def main():
    cameras = read_camera_parameters(r"10\templeRing\templeR_par.txt")
    all_points_3D = []
    for i in range(len(cameras) - 1):
        print(f"Processing image pair {i+1}/{len(cameras)-1}")
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
        distances = np.sqrt(np.sum(points_3D**2, axis=1))
        mask = distances < np.percentile(distances, 95)
        points_3D = points_3D[mask]
        all_points_3D.append(points_3D)
    cv2.destroyAllWindows()
    all_points_3D = np.vstack(all_points_3D)
    fig = px.scatter_3d(
        x=all_points_3D[:, 0],
        y=all_points_3D[:, 1],
        z=all_points_3D[:, 2],
        title="3D Reconstruction from Multiple Views",
    )
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()
    fig.write_html("10/templeR_multi.html")


if __name__ == "__main__":
    main()
