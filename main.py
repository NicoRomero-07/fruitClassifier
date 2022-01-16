import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

cv2.namedWindow("Fruit Classifier")
vc = cv2.VideoCapture(0);

while True:
    next, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    size_sobel = 3
    const_k = 0.04
    size_window = 2

    harris = cv2.cornerHarris(gray, size_window, size_sobel, const_k)
    threshold = 0.01 * harris.max()
    _, corners = cv2.threshold(harris, threshold, harris.max(), cv2.THRESH_BINARY)

    image = cv2.Canny(frame, 100, 200, apertureSize=3)

    cv2.imshow("Fruit Classifier", image)
    if cv2.waitKey(50) >= 0:
        break
