import math
import numpy as np
import cv2


def calcular_angulos(imagem, tamanho_bloco):
    """
    anisotropy orientation estimate, based on equations 5 from:
    https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf
    :param imagem:
    :param tamanho_bloco: int width of the ridge
    :return: array
    """
    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2

    (y, x) = imagem.shape

    matriz_sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = np.array(matriz_sobel).astype(np.int)
    sobel_x = np.transpose(sobel_y).astype(np.int)

    retorno = [[] for i in range(1, y, tamanho_bloco)]

    gx_ = cv2.filter2D(imagem / 125, -1, sobel_y) * 125
    gy_ = cv2.filter2D(imagem / 125, -1, sobel_x) * 125

    for j in range(1, y, tamanho_bloco):
        for i in range(1, x, tamanho_bloco):
            nominador = 0
            denominador = 0
            for l in range(j, min(j + tamanho_bloco, y - 1)):
                for k in range(i, min(i + tamanho_bloco, x - 1)):
                    gx = round(gx_[l, k])  # horizontal gradients at l, k
                    gy = round(gy_[l, k])  # vertial gradients at l, k
                    nominador += j1(gx, gy)
                    denominador += j2(gx, gy)

            if nominador or denominador:
                angulo = (math.pi + math.atan2(nominador, denominador)) / 2
                orientation = np.pi / 2 + math.atan2(nominador, denominador) / 2
                retorno[int((j - 1) // tamanho_bloco)].append(angulo)
            else:
                retorno[int((j - 1) // tamanho_bloco)].append(0)

    retorno = np.array(retorno)

    return retorno


def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def obter_kernel(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel


def encontrar_limites_linhas(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W / 2) * tang + j + W / 2))
        end = (i + W, int((W / 2) * tang + j + W / 2))
    else:
        begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
        end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)
    return (begin, end)


def gerar_imagem_angulos(im, mask, angles, W):
    (y, x) = im.shape
    result = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    mask_threshold = (W - 1) ** 2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i - 1:i + W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = encontrar_limites_linhas(i, j, W, tang)
                cv2.line(result, begin, end, color=[0, 150, 0])

    cv2.resize(result, im.shape, result)
    return result
