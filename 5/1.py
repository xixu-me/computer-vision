import cv2 as cv

image = cv.imread("images/lena.jpg")

sift = cv.SIFT_create()  # type: ignore

keypoints, descriptors = sift.detectAndCompute(image, None)

image_with_keypoints = cv.drawKeypoints(image, keypoints, None)  # type: ignore

cv.imshow("SIFT Keypoints", image_with_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()
