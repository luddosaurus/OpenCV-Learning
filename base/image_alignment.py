import cv2 as cv2
import numpy as np
from base.utils.find_contours import *
from base.utils.stack_images import *

built_in_cam = 0
external_cam = 1
cam = built_in_cam
frameWidth = 640
frameHeight = 480

imgWidth = 640
imgHeight = 480

cap = cv2.VideoCapture(cam)
cap.set(3, frameHeight)
cap.set(4, frameWidth)
cap.set(10, 150)


def pre_processing(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(src=imgGray, ksize=(5, 5), sigmaX=1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel=kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    return imgThreshold


def get_warp(image, biggest_box_points, img_height, img_width):
    ordered_box_points = reorder(biggest_box_points)
    src_points = np.float32(ordered_box_points)
    dst_points = np.float32([
        [0, 0],
        [img_width, 0],
        [0, img_height],
        [img_width, img_height]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(image, matrix, (img_width, img_height))
    return img_output


def reorder(points):
    points = points.reshape((4, 2))
    reshaped_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)

    reshaped_points[0] = points[np.argmin(add)]
    reshaped_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reshaped_points[1] = points[np.argmin(diff)]
    reshaped_points[2] = points[np.argmax(diff)]
    return reshaped_points


while True:
    success, img = cap.read()
    imgContour = img.copy()
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgProcessed = pre_processing(img)
    biggestBox = get_contours2(imgProcessed, imgContour)

    if biggestBox.size != 0:
        imgWarped = get_warp(img, biggestBox, imgHeight, imgWidth)
        imageArray = ([imgContour, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imageArray = ([imgContour, img])

    stackedImages = stack_images(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
