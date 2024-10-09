import cv2 as cv

big = cv.imread("images/big.jpg")
small = cv.imread("images/small.jpg")
cv.imshow("big", big)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("small", small)
cv.waitKey(0)
cv.destroyAllWindows()

sift = cv.SIFT_create()  # type: ignore

kp1, des1 = sift.detectAndCompute(big, None)
kp2, des2 = sift.detectAndCompute(small, None)

bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)  # type: ignore

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

result = cv.drawMatches(big, kp1, small, kp2, matches[:15], None)  # type: ignore

cv.imshow("-match", result)
cv.waitKey(0)
cv.destroyAllWindows()
