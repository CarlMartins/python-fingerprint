import math
import numpy as np
import cv2 as cv


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

    sobel_operator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    y_sobel = np.array(sobel_operator).astype(np.int)
    x_sobel = np.transpose(y_sobel).astype(np.int)

    result = [[] for i in range(1, y, tamanho_bloco)]

    gx_ = cv.filter2D(imagem / 125, -1, y_sobel) * 125
    gy_ = cv.filter2D(imagem / 125, -1, x_sobel) * 125

    for j in range(1, y, tamanho_bloco):
        for i in range(1, x, tamanho_bloco):
            nominator = 0
            denominator = 0
            for l in range(j, min(j + tamanho_bloco, y - 1)):
                for k in range(i, min(i + tamanho_bloco, x - 1)):
                    Gx = round(gx_[l, k])  # horizontal gradients at l, k
                    Gy = round(gy_[l, k])  # vertial gradients at l, k
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)

            # nominator = round(np.sum(gy_[j:min(j + tamanho_bloco, y - 1), i:min(i + tamanho_bloco , tom_pixel - 1)]))
            # denominator = round(np.sum(gx_[j:min(j + tamanho_bloco, y - 1), i:min(i + tamanho_bloco , tom_pixel - 1)]))
            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                orientation = np.pi / 2 + math.atan2(nominator, denominator) / 2
                result[int((j - 1) // tamanho_bloco)].append(angle)
            else:
                result[int((j - 1) // tamanho_bloco)].append(0)

            # segment image
            # focus_img = imagem[j:min(j + tamanho_bloco, y - 1), i:min(i + tamanho_bloco , tom_pixel - 1)]
            # segmentator = -1 if segmentator/tamanho_bloco*tamanho_bloco < np.max(focus_img)*

    result = np.array(result)

    return result


def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel


def smooth_angles(angles):
    """
    reference: https://airccj.org/CSCP/vol7/csit76809.pdf pg91
    Practically, it is possible to have a block so noisy that the directional estimate is completely false.
    This then causes a very large angular variation between two adjacent blocks. However, a
    fingerprint has some directional continuity, such a variation between two adjacent blocks is then
    representative of a bad estimate. To eliminate such discontinuities, a low-pass filter is applied to
    the directional board.
    :param angles:
    :return:
    """
    angles = np.array(angles)
    cos_angles = np.cos(angles.copy() * 2)
    sin_angles = np.sin(angles.copy() * 2)

    kernel = np.array(kernel_from_function(5, gauss))

    cos_angles = cv.filter2D(cos_angles / 125, -1, kernel) * 125
    sin_angles = cv.filter2D(sin_angles / 125, -1, kernel) * 125
    smooth_angles = np.arctan2(sin_angles, cos_angles) / 2

    return smooth_angles


def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W / 2) * tang + j + W / 2))
        end = (i + W, int((W / 2) * tang + j + W / 2))
    else:
        begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
        end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)
    return (begin, end)


def gerar_imagem_angulos(im, mask, angles, W):
    (y, x) = im.shape
    result = cv.cvtColor(np.zeros(im.shape, np.uint8), cv.COLOR_GRAY2RGB)
    mask_threshold = (W - 1) ** 2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i - 1:i + W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = get_line_ends(i, j, W, tang)
                cv.line(result, begin, end, color=150)

    cv.resize(result, im.shape, result)
    return result
