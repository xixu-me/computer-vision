import cv2 as cv


def build_gaussian_pyramid(image, octaves, scales):
    gaussian_pyramid = []
    for o in range(octaves):
        octave_images = []
        for s in range(scales):
            if s == 0:
                if o == 0:
                    octave_images.append(image)
                else:
                    octave_images.append(cv.pyrDown(gaussian_pyramid[o - 1][-3]))
            else:
                sigma = 1.6 * (2 ** (s / scales))
                blurred = cv.GaussianBlur(octave_images[0], (5, 5), sigma)
                octave_images.append(blurred)

        gaussian_pyramid.append(octave_images)
    return gaussian_pyramid


def build_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_images = []
        for s in range(len(octave) - 1):
            dog_images.append(octave[s + 1] - octave[s])
        dog_pyramid.append(dog_images)
    return dog_pyramid


image = cv.imread("images/lena.jpg", cv.IMREAD_GRAYSCALE)
guassian_pyramid = build_gaussian_pyramid(image, octaves=4, scales=5)
dog_pyramid = build_dog_pyramid(guassian_pyramid)

for i in range(len(dog_pyramid)):
    for j in range(len(dog_pyramid[i])):
        cv.namedWindow("DoG pyramid image %d %d" % (i, j))
        cv.imshow("DoG pyramid image %d %d" % (i, j), dog_pyramid[i][j])

cv.waitKey(0)
cv.destroyAllWindows()
