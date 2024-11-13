import cv2
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

img1 = cv2.imread(r"10\templeRing\templeR0003.png")
img2 = cv2.imread(r"10\templeRing\templeR0005.png")

fx = 0.25 * 1520.4
fy = 0.25 * 1525.9
cx = 302.32
cy = 246.87
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

R1 = np.array(
    [
        [-0.01625331773280620100, 0.98386957700862299000, -0.17814736905031653000],
        [0.97668439268305030000, -0.02252259937820530100, -0.21349550254417543000],
        [-0.21406407160478280000, -0.17746376518636725000, -0.96056399333613396000],
    ]
)
t1 = np.array([[-0.0283090812583], [-0.0366442193256], [0.529139415773]])

R2 = np.array(
    [
        [-0.05235090589954815400, 0.98479784589115438000, -0.16562785206491965000],
        [0.88539496349116698000, -0.03093875352325572600, -0.46380874523331522000],
        [-0.46188217250287084000, -0.17092687400923678000, -0.87031538103463302000],
    ]
)
t2 = np.array([[-0.0269600886818], [-0.0469344855587], [0.53860946783]])

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

points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
points_3D = points_4D / points_4D[3]
points_3D = points_3D[:3, :].T

distances = np.sqrt(np.sum(points_3D**2, axis=1))
mask = distances < np.percentile(distances, 95)
points_3D = points_3D[mask]

fig = px.scatter_3d(
    x=points_3D[:, 0], y=points_3D[:, 1], z=points_3D[:, 2], title="3D Reconstruction"
)

fig.update_traces(marker=dict(size=1))
fig.update_layout(scene=dict(aspectmode="data"))

fig.show()

fig.write_html("10/templeR.html")

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
cv2.waitKey(0)
cv2.destroyAllWindows()
