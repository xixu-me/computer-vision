import cv2 as cv
import numpy as np

img = cv.imread("images/lena.png")
cv.imshow("raw_img", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

J = (0.05, 0.01, 0.005)

for j in J:
    dst = cv.cornerHarris(gray, 2, 3, 0.04)  # type: ignore
    a = dst > j * dst.max()
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if a[r, c] != 0:
                cv.circle(img, (c, r), 1, (0, 0, 255), 1)
    cv.imshow("corners_" + str(j), img)
    cv.waitKey(0)

cv.waitKey(0)
cv.destroyAllWindows()
