import cv2 as cv

img1 = cv.imread("images/big.jpg", 0)
img2 = cv.imread("images/small.jpg", 0)

sift = cv.SIFT_create()  # type: ignore

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

matches_all = []
for m, n in matches:
    matches_all.append(m)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

img_matches_without_rs = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches_all, None, flags=2)  # type: ignore
img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)  # type: ignore

cv.imshow("Match-ori", img_matches_without_rs)
cv.imshow("Matches", img_matches)
cv.waitKey(0)
cv.destroyAllWindows()
