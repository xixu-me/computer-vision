# 用OpenCV库中的均值漂移函数cv2.pyrMeanShiftFiltering()来实现图像分割，需要先将图像从BGR转换到YUV颜色空间，分割后将分割结果从YUV转换回BGR颜色空间显示。

import cv2 as cv
import numpy as np


def mean_shift_segmentation(image, spatial_radius, color_radius):
    image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    segmented_image = cv.pyrMeanShiftFiltering(image_yuv, spatial_radius, color_radius)
    return cv.cvtColor(segmented_image, cv.COLOR_YUV2BGR)


image = cv.imread("images/lena-color.png")
spatial_radius = 10
color_radius = 10
segmented_image = mean_shift_segmentation(image, spatial_radius, color_radius)

cv.imshow("Original Image", image)
cv.imshow("Mean Shift Segmented Image", segmented_image)
cv.waitKey(0)
cv.destroyAllWindows()
