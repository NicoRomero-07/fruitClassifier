import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

images_path = './images/'
fruits = ["apples", "bananas"]

descriptor_bananas = np.zeros((14, 7))
descriptor_apples = np.zeros((14, 7))
descriptors = np.array([descriptor_apples, descriptor_bananas])


def genDescriptor(image, imageBbRgb):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    xMax, yMax, wMax, hMax = 0, 0, 0, 0
    maxCnt = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(h) * abs(w) > abs(hMax) * abs(wMax):
            xMax, yMax, wMax, hMax = x, y, w, h
            maxCnt = cnt
    descriptor = np.array([0., 0., 0., 0., 0., 0., 0.])
    area = float(cv2.contourArea(maxCnt))
    perimetro = float(cv2.arcLength(maxCnt, True))
    descriptor[0] = area
    descriptor[1] = perimetro
    descriptor[2] = 4 * np.pi * (area / perimetro ** 2)
    (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(maxCnt)
    descriptor[3] = np.sqrt(majorAxisLength ** 2 - minorAxisLength ** 2) / majorAxisLength
    descriptor[4] = np.mean(imageBbRgb[:, :, 0])
    descriptor[5] = np.mean(imageBbRgb[:, :, 1])
    descriptor[6] = np.mean(imageBbRgb[:, :, 2])

    return descriptor


for f in range(0, 2):
    for i in range(1, 15):
        image = cv2.imread(images_path + "binarized/" + str(fruits[f]) + "/" + str(i) + ".jpg", 0)
        imageBbRgb = cv2.imread(images_path + "rgbBoundingBox/" + str(fruits[f]) + "/" + str(i) + ".jpg",
                                cv2.IMREAD_COLOR)
        descriptors[f, i - 1, :] = genDescriptor(image, imageBbRgb)

print("APPLES: ", descriptors[0])
print("BANANAS: ", descriptors[1])
np.save("./data/descriptor_apples.npy", descriptors[0, :, :])
np.save("./data/descriptor_bananas.npy", descriptors[1, :, :])
