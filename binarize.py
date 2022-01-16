import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

images_path = './images/'
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
cv2.setRNGSeed(5)

descriptor_apples = np.zeros((14, 7))
descriptor_bananas = np.zeros((14, 7))
descriptors = np.array([descriptor_apples, descriptor_bananas])
fruits = ["apples", "bananas"]


def computeImage(imageRGB):
    image = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 5)
    kernel = np.ones((5, 5), np.uint8)
    ret, binarizedOtsu = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    closing = cv2.morphologyEx(binarizedOtsu, cv2.MORPH_CLOSE, kernel)
    dilating = cv2.dilate(closing, kernel)
    erosing = cv2.erode(dilating, kernel)

    contours, hierarchy = cv2.findContours(erosing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    xMax, yMax, wMax, hMax = 0, 0, 0, 0
    maxCnt = 0
    for cnt in contours:
        print(idx)
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(h) * abs(w) > abs(hMax) * abs(wMax):
            xMax, yMax, wMax, hMax = x, y, w, h
            maxCnt = cnt

    if idx > 0:
        final = erosing[yMax:yMax + hMax, xMax:xMax + wMax]
    else:
        final = erosing

    final = cv2.resize(final, (400, 500))
    imageRGBbb = imageRGB[yMax:yMax + hMax, xMax:xMax + wMax]
    cv2.imwrite("./images/rgbBoundingBox/" + fruits[f] + "/" + str(i) + ".jpg", imageRGB)
    return final, imageRGBbb


for f in range(0, 2):
    for i in range(1, 15):
        imageRGB = cv2.imread(images_path + "/rgb/" + str(fruits[f]) + "/" + str(i) + ".jpg", cv2.IMREAD_COLOR)

        finalImage, x = computeImage(imageRGB)
        cv2.imwrite("./images/binarized/" + fruits[f] + "/" + str(i) + ".jpg", finalImage)
