import cv2
import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size):
    """
    https://airccj.org/CSCP/vol7/csit76809.pdf pg93
    Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

    :param pixels:
    :param i:
    :param j:
    :return:
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
                     (0, 1), (1, 1), (1, 0),  # p8    p4
                     (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),  # p1 p2   p3
                     (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),  # p8      p4
                     (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]  # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values) - 1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"


def calculate_minutiaes(img, imgSkel, freq, limiteLinha, limiteColun, kernel_size=3):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    biniry_image = np.zeros_like(imgSkel)
    biniry_image[imgSkel < 10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = imgSkel.shape
    result = cv.cvtColor(imgSkel, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}
    coordenadas_minutias = []

    # iterate each pixel minutia
    for i in range(1, x - kernel_size // 2):
        for j in range(1, y - kernel_size // 2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none" and verificaBorda(freq, j, i) != True and limiteLinha[0] + 5 < j < limiteLinha[1] - 5\
                    and (limiteColun[0] + 5 < i < limiteColun[1] - 5):
                coordenadas_minutias.append(cv2.KeyPoint(i, j, 1))
                cv.circle(result, (i, j), radius=5, color=colors[minutiae], thickness=1)
                cv.circle(img, (i, j), radius=6, color=colors[minutiae], thickness=1)

    return img, result, coordenadas_minutias


def verificaBorda(freq, j, i):
    for x in range(j - 10, j + 10):
        for y in range(i - 10, i + 10):
            if freq[x][y] == 0:
                return True
    return False
