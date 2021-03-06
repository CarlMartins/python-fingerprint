import cv2
import cv2 as cv
import numpy as np


def verifica_minutia(pixels, i, j, tamanho_kernel):
    # Se o pixel testado é preto, é uma crista da digital
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

        # Conta a quantidade de pixels pretos ao redor do pixel testado
        pixels_pretos = 0
        for pixel in range(0, len(valores_pixels) - 1):
            pixels_pretos += abs(valores_pixels[pixel] - valores_pixels[pixel + 1])
        pixels_pretos //= 2

        # Se só existir um pixel preto ao redor do pixel testado, então é uma ponta
        # Se existirem 3 pixels pretos ao redor do pixel testado, então é uma bifurcação
        if pixels_pretos == 1:
            return "ponta"
        if pixels_pretos == 3:
            return "bifurcacao"

    return "none"


def identificar_minucias(imagem, img_esqueleto, mapa_frequencia, limite_linha, limite_coluna, tamanho_bloco=5):
    imagem = cv.cvtColor(imagem, cv.COLOR_GRAY2RGB)

    imagem_binaria = np.zeros_like(img_esqueleto)
    imagem_binaria[img_esqueleto < 10] = 1.0
    imagem_binaria = imagem_binaria.astype(np.int8)

    (y, x) = img_esqueleto.shape
    resultado = cv.cvtColor(img_esqueleto, cv.COLOR_GRAY2RGB)
    cores = {"ponta": (150, 0, 0), "bifurcacao": (0, 150, 0)}
    coordenadas_minucias = []

    # Iterar por cada pixel em busca de minúcias
    for i in range(1, x - tamanho_bloco // 2):
        for j in range(1, y - tamanho_bloco // 2):
            minucia = verifica_minutia(imagem_binaria, j, i, tamanho_bloco)
            if minucia != "none" and verifica_borda(mapa_frequencia, j, i) is not True and limite_linha[0] + 5 < j < \
                    limite_linha[1] - 5 \
                    and (limite_coluna[0] + 5 < i < limite_coluna[1] - 5):
                # if minucia != "none":
                coordenadas_minucias.append(cv2.KeyPoint(i, j, 1))
                cv.circle(resultado, (i, j), radius=5, color=cores[minucia], thickness=2)
                cv.circle(imagem, (i, j), radius=6, color=cores[minucia], thickness=2)

    return imagem, resultado, coordenadas_minucias


def verifica_borda(freq, j, i):
    for x in range(j - 10, j + 10):
        for y in range(i - 10, i + 10):
            if freq[x][y] == 0:
                return True
    return False
