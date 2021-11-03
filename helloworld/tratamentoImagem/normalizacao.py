from math import sqrt
import numpy as np
import cv2


def normalizar_pixel(tom_pixel, variancia_alvo, variancia_imagem, tom_medio, tom_medio_alvo):
    coeficiente_media = sqrt((variancia_alvo * ((tom_pixel - tom_medio) ** 2)) / variancia_imagem)
    return tom_medio_alvo + coeficiente_media if tom_pixel > tom_medio else tom_medio_alvo - coeficiente_media


def normalizar(imagem, tom_medio_alvo, variacao_alvo):
    tom_medio = np.mean(imagem)
    variancia_imagem = np.std(imagem) ** 2
    (linha, coluna) = imagem.shape
    imagem_normalizada = imagem.copy()
    for col in range(coluna):
        for lin in range(linha):
            imagem_normalizada[lin, col] = normalizar_pixel(imagem[lin, col], variacao_alvo, variancia_imagem,
                                                            tom_medio, tom_medio_alvo)

    return imagem_normalizada
