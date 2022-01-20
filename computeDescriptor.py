import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import functions

images_path = './images/'
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
cv2.setRNGSeed(5)

descriptor_apples = np.zeros((14, 5))
descriptor_bananas = np.zeros((14, 5))
descriptors = np.array([descriptor_apples, descriptor_bananas])
fruits = ["apples", "bananas"]

for f in range(0, np.shape(fruits)[0]):
    for i in range(1, 15):
        imageRGB = cv2.imread(images_path + "/rgb/" + str(fruits[f]) + "/" + str(i) + ".jpg", cv2.IMREAD_COLOR)
        finalImage, x = functions.computeImage(imageRGB)
        descriptors[f, i - 1, :] = functions.genDescriptor(finalImage, x)
        cv2.imwrite("./images/binarized/" + fruits[f] + "/" + str(i) + ".jpg", finalImage)

print("APPLES: ", descriptors[0])
print("BANANAS: ", descriptors[1])
np.save("./data/descriptor_apples.npy", descriptors[0, :, :])
np.save("./data/descriptor_bananas.npy", descriptors[1, :, :])
