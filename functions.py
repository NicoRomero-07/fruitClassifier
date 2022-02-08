import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats


def image_moments(region):
    """ Compute moments of the external contour in a binary image.

        Args:
            region: Binary image

        Returns:
            moments: dictionary containing all moments of the region
    """

    # Get external contour
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    # Compute moments
    moment = cv2.moments(cnt)

    return moment


def genDescriptor(image, imageBbRgb):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    xMax, yMax, wMax, hMax = 0, 0, 0, 0
    maxCnt = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(hMax) * abs(wMax) < abs(h) * abs(w):
            xMax, yMax, wMax, hMax = x, y, w, h
            maxCnt = cnt

    descriptor = np.array([0., 0., 0., 0., 0., 0., 0.])
    area = float(cv2.contourArea(maxCnt))
    perimetro = float(cv2.arcLength(maxCnt, True))
    descriptor[0] = perimetro ** 2 / area
    descriptor[1] = 4 * np.pi * (area / perimetro ** 2)

    imageHSV = cv2.cvtColor(imageBbRgb, cv2.COLOR_BGR2HSV)

    descriptor[2] = np.mean(imageHSV[:, :, 0])
    descriptor[3] = np.mean(imageHSV[:, :, 1])
    descriptor[4] = np.mean(imageHSV[:, :, 2])

    moments = image_moments(image)
    huMoments = cv2.HuMoments(moments).flatten()
    descriptor[5] = huMoments[0]
    descriptor[6] = huMoments[1]

    return descriptor


def computeImage(imageRGB):
    erosing = computeSameSize(imageRGB)
    contours, hierarchy = cv2.findContours(erosing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    xMax, yMax, wMax, hMax = 0, 0, 0, 0
    maxCnt = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(hMax) * abs(wMax) < abs(h) * abs(w) < (np.shape(erosing)[0] * np.shape(erosing)[1]):
            xMax, yMax, wMax, hMax = x, y, w, h
            maxCnt = cnt

    if idx > 0:
        final = erosing[yMax:yMax + hMax, xMax:xMax + wMax]
    else:
        final = erosing

    final = cv2.resize(final, (400, 500))
    imageRGBbb = imageRGB[yMax:yMax + hMax, xMax:xMax + wMax]
    imageRGBbb = cv2.resize(imageRGBbb, (400, 500))

    return final, imageRGBbb


def resize(image):
    scale_percent = 220  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def computeSameSize(imageRGB):
    image = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 5)
    kernel = np.ones((5, 5), np.uint8)
    ret, binarizedOtsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(binarizedOtsu, cv2.MORPH_CLOSE, kernel)
    dilating = cv2.dilate(closing, kernel)
    erosing = cv2.erode(dilating, kernel)

    return erosing
