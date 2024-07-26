import cv2 as cv
import numpy as np


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_images_from_disk(pixel):

    X = []
    Y = []
    print("preprocessing images cats ")
    for i in range(1, 4000):
        img = cv.imread('cats\\cat.' + str(i) + '.jpg')
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.resize(img_gray, (pixel, pixel), interpolation=cv.INTER_CUBIC)

        res = np.ndarray.flatten(res) / 255

        X.append(res)
        Y.append(1)

    print("preprocessing images dogs ")
    for i in range(1, 4000):
        img = cv.imread('dogs\\dog.' + str(i) + '.jpg')
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.resize(img_gray, (pixel, pixel), interpolation=cv.INTER_CUBIC)

        res = np.ndarray.flatten(res) / 255

        X.append(res)
        Y.append(0)


    X = np.asarray(X)
    Y = np.asarray(Y).reshape((7998,1))

    X_new, Y_new = unison_shuffled_copies(X,Y)

    return X_new, Y_new






























