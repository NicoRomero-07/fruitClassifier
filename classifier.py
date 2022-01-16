import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats

images_path = './images/test/'
fruits = ["apples", "bananas"]

train_apples = np.load("./data/descriptor_apples.npy")[:, :].T
train_bananas = np.load("./data/descriptor_bananas.npy")[:, :].T

# Compute covariance matrices
cov_apples = np.cov(train_apples)
cov_bananas = np.cov(train_bananas)

# Compute means
mean_apples = np.mean(train_apples, axis=1)
mean_bananas = np.mean(train_bananas, axis=1)


def discriminant_function(features, mu, cov, prior):
    """ Evaluates the discriminant function d(x)

        Args:
            features: feature vector of dimension n
            mu: mean vector of the class of which is being computed the probability
            cov: covariance matrix with shape (n,n) of the class
            prior: prior probability of class k

        Returns:
            dx: result of discriminant function
    """
    covinv = np.linalg.inv(cov)  # Auxiliar variable
    muTraspuesta = np.transpose(mu)
    featuresTrapuesta = np.transpose(features)

    matrizaux = np.dot(muTraspuesta, covinv)
    segundaParteIndTerm = np.log(np.linalg.det(cov)) + np.dot(matrizaux, mu)
    indpTerm = np.log(prior) - (1 / 2) * segundaParteIndTerm

    matrizaux1 = np.dot(featuresTrapuesta, covinv)
    linear = np.dot(matrizaux1, mu)

    matrizaux2 = np.dot(featuresTrapuesta, covinv)
    quadratic = (-1 / 2) * (np.dot(matrizaux2, features))

    dx = indpTerm + linear + quadratic  # You can divide this computation in as many lines as you need

    return dx


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

    return final, imageRGBbb


def genDescriptor(image, imageBbRgb):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    xMax, yMax, wMax, hMax = 0, 0, 0, 0
    maxCnt = 0
    for cnt in  contours:
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


def classify_image(sign_image):
    """ Classify a traffic sign image by its shape using a bayesian classifier

        Args:
            sign_image: Binarized image
    """

    # Compute descriptor
    image, imageRGBbb = computeImage(sign_image)
    descriptor = genDescriptor(image, imageRGBbb)
    # Classify circle test image
    prior = 1 / 2
    apple = discriminant_function(descriptor, mean_apples, cov_apples, prior)
    print(apple)
    banana = discriminant_function(descriptor, mean_bananas, cov_bananas, prior)
    print(banana)

    # Search the maximum
    classification = max([apple, banana])

    if classification == apple:
        print("The sign is a apple\n")
    elif classification == banana:
        print("The sign is a banana\n")
    else:
        print("The sign is a error\n")

    return descriptor


# Read images
test_apple = cv2.imread(images_path + "test_apple.jpg", cv2.IMREAD_COLOR)
test_banana = cv2.imread(images_path + "test_banana.jpg", cv2.IMREAD_COLOR)

# Classify them
print("Apple: ")
descriptor_apple = classify_image(test_apple)
print("Banana: ")
descriptor_banana = classify_image(test_banana)
