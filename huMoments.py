import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

images_path = './images/binarized/'


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
    moments = cv2.moments(cnt)

    return moments


hu_apples = np.zeros((14, 7))

for i in range(1, 15):
    img = cv2.imread(images_path + "apples/" + str(i) + ".jpg", 0)
    print(i)
    moments = image_moments(img)
    hu_apples[i - 1, :] = cv2.HuMoments(moments).flatten()

hu_bananas = np.zeros((14, 7))

for i in range(1, 15):
    img = cv2.imread(images_path + "bananas/" + str(i) + ".jpg", 0)
    moments = image_moments(img)
    hu_bananas[i - 1, :] = cv2.HuMoments(moments).flatten()

# Define plot axis
plt.axis([-0.1, 0.8, -0.0065, 0.0285])

# Plot firsts two Hu moments
plt.xlabel("First Hu moment")
plt.ylabel("Second Hu moment")
plt.scatter(hu_apples[:, 0], hu_apples[:, 1], marker="^")
plt.scatter(hu_bananas[:, 0], hu_bananas[:, 1], marker="o")
plt.show()
