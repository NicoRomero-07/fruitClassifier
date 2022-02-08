import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import functions
import fruitsValues

cv2.namedWindow("Fruit Classifier")
vc = cv2.VideoCapture(0)

# Leer transmisión de video
# vc = cv2.VideoCapture(1)

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


def classify_image(sign_image):
    """ Classify a traffic sign image by its shape using a bayesian classifier

        Args:
            sign_image: Binarized image
    """

    # Compute descriptor
    image, imageRGBbb = functions.computeImage(sign_image)
    descriptor = functions.genDescriptor(image, imageRGBbb)
    print(descriptor)
    # Classify circle test image
    prior = 1 / len(fruits)
    apple = discriminant_function(descriptor, mean_apples, cov_apples, prior)
    print(apple)
    banana = discriminant_function(descriptor, mean_bananas, cov_bananas, prior)
    print(banana)

    # Search the maximum
    classification = max([apple, banana])

    f = fruitsValues.unDefine()
    if classification == apple:
        print("The sign is a apple\n")
        f = fruitsValues.Apple()
    elif classification == banana:
        print("The sign is a banana\n")
        f = fruitsValues.Banana()
    else:
        print("The sign is a error\n")

    return descriptor, f


texto = "?"

while True:
    while True:
        next, frame = vc.read()
        cv2.imshow("Fruit Classifier", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.imwrite('./imageToCompute.jpg', frame)
    imageToCompute = cv2.imread("./imageToCompute.jpg", cv2.IMREAD_COLOR)
    _, fruta = classify_image(imageToCompute)

    font = cv2.FONT_HERSHEY_TRIPLEX
    tamanoLetra = 2
    grosorLetra = 3

    while True:
        next2, frame2 = vc.read()
        imageRGB = frame2
        image = functions.computeSameSize(imageRGB)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        xMax, yMax, wMax, hMax = 0, 0, 0, 0
        maxCnt = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(hMax) * abs(wMax) < abs(h) * abs(w) < (np.shape(image)[0] * np.shape(image)[1]):
                xMax, yMax, wMax, hMax = x, y, w, h
                maxCnt = cnt

        colors = (0, 0, 0)

        fruitBoundingBox = image[xMax:xMax + wMax, yMax:yMax + hMax]
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(imageRGB, (xMax, yMax), (xMax + wMax, yMax + hMax), colors, 5)
        cv2.drawContours(imageRGB, maxCnt, -1, (0, 0, 255), 2, cv2.LINE_AA)

        # Escribir texto
        ubicacionNombre = (xMax, yMax)
        # Nombre
        cv2.putText(imageRGB, fruta.name, ubicacionNombre, font, tamanoLetra, fruta.color, grosorLetra)
        # Valor nutricional
        d = 60
        cv2.putText(imageRGB, "Kcal: " + str(fruta.kcal), (xMax + wMax - fruta.d, yMax), font, 1, fruta.color, 1)
        cv2.putText(imageRGB, "Proteins: " + str(fruta.proteins), (xMax + wMax - fruta.d, yMax - 25), font, 1,
                    fruta.color, 1)
        cv2.putText(imageRGB, "Hydrates: " + str(fruta.hydrates), (xMax + wMax - fruta.d, yMax - 50), font, 1,
                    fruta.color, 1)
        cv2.putText(imageRGB, "Fat: " + str(fruta.fat), (xMax + wMax - fruta.d, yMax - 75), font, 1, fruta.color, 1)
        cv2.imshow("Fruit Classifier", imageRGB)

        if cv2.waitKey(50) >= 0:
            break
