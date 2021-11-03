"""
In order to eliminate the edges of the image and areas that are too noisy, segmentation is
necessary. It is based on the calculation of the variance of gray levels. For this purpose, the image
is divided into sub-blocks of (tamanho_bloco × tamanho_bloco) size’s and for each block the variance.
Then, the root of the variance of each block is compared with a threshold T, if the value obtained
is lower than the threshold, then the corresponding block is considered as the background of the
image and will be excluded by the subsequent processing.

The selected threshold value is T = 0.2 and the selected block size is tamanho_bloco = 16

This step makes it possible to reduce the size of the useful part of the image and subsequently to
optimize the extraction phase of the biometric data.
"""

import numpy as np
import cv2 as cv


def normalizar_img_segmentada(img):
    return (img - np.mean(img)) / (np.std(img))


def segmentar_imagem(imagem, tamanho_bloco, threshold=.2):
    """
    Returns mascara identifying the ROI. Calculates the standard deviation in each image block and threshold the ROI
    It also normalises the intesity values of
    the image so that the ridge regions have zero mean, unit standard
    deviation.
    :param imagem: Image
    :param tamanho_bloco: size of the block
    :param threshold: std threshold
    :return: imagem_segmentada
    """
    (linha, coluna) = imagem.shape
    threshold = np.std(imagem) * threshold

    variancia_imagem = np.zeros(imagem.shape)
    imagem_segmentada = imagem.copy()
    mascara = np.ones_like(imagem)

    for col in range(0, coluna, tamanho_bloco):
        for lin in range(0, linha, tamanho_bloco):
            bloco = [col, lin, min(col + tamanho_bloco, coluna), min(lin + tamanho_bloco, linha)]
            variancia_bloco = np.std(imagem[bloco[1]:bloco[3], bloco[0]:bloco[2]])
            variancia_imagem[bloco[1]:bloco[3], bloco[0]:bloco[2]] = variancia_bloco

    # apply threshold
    mascara[variancia_imagem < threshold] = 0

    # smooth mascara with a open/close morphological filter
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (tamanho_bloco * 2, tamanho_bloco * 2))
    mascara = cv.morphologyEx(mascara, cv.MORPH_OPEN, kernel)
    mascara = cv.morphologyEx(mascara, cv.MORPH_CLOSE, kernel)

    # normalizar segmented image
    imagem_segmentada *= mascara
    imagem = normalizar_img_segmentada(imagem)
    valor_medio = np.mean(imagem[mascara == 0])
    valor_variancia = np.std(imagem[mascara == 0])
    img_seg_normalizada = (imagem - valor_medio) / valor_variancia

    return imagem_segmentada, img_seg_normalizada, mascara
