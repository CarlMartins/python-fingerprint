import cv2
import cv2 as cv
import numpy as np


def verifica_minutia(pixels, i, j, tamanho_kernel):
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
    :tamanho_kernel:
    :return:
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if tamanho_kernel == 3:
            celulas = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
                     (0, 1), (1, 1), (1, 0),  # p8    p4
                     (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5
        else:
            celulas = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),  # p1 p2   p3
                     (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),  # p8      p4
                     (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]  # p7 p6   p5

        valores_pixels = [pixels[i + l][j + k] for k, l in celulas]

        # count crossing how many times it goes from 0 to 1
        pixels_pretos = 0
        for pixel in range(0, len(valores_pixels) - 1):
            pixels_pretos += abs(valores_pixels[pixel] - valores_pixels[pixel + 1])
        pixels_pretos //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if pixels_pretos == 1:
            return "ending"
        if pixels_pretos == 3:
            return "bifurcation"

    return "none"


def identificar_minucias(imagem, img_esqueleto, freq, limite_linha, limite_coluna, tamanho_bloco=5):
    imagem = cv.cvtColor(imagem, cv.COLOR_GRAY2RGB)

    imagem_binaria = np.zeros_like(img_esqueleto)
    imagem_binaria[img_esqueleto < 10] = 1.0
    imagem_binaria = imagem_binaria.astype(np.int8)

    (y, x) = img_esqueleto.shape
    result = cv.cvtColor(img_esqueleto, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}
    coordenadas_minutias = []

    # iterate each pixel minutia
    for i in range(1, x - tamanho_bloco // 2):
        for j in range(1, y - tamanho_bloco // 2):
            minutiae = verifica_minutia(imagem_binaria, j, i, tamanho_bloco)
            if minutiae != "none" and verificaBorda(freq, j, i) != True and limite_linha[0] + 5 < j < limite_linha[1] - 5\
                    and (limite_coluna[0] + 5 < i < limite_coluna[1] - 5):
                coordenadas_minutias.append(cv2.KeyPoint(i, j, 1))
                cv.circle(result, (i, j), radius=5, color=colors[minutiae], thickness=2)
                cv.circle(imagem, (i, j), radius=6, color=colors[minutiae], thickness=2)

    return imagem, result, coordenadas_minutias


def verificaBorda(freq, j, i):
    for x in range(j - 10, j + 10):
        for y in range(i - 10, i + 10):
            if freq[x][y] == 0:
                return True
    return False
