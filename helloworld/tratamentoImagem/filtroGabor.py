import numpy as np
import scipy
import cv2


def filtro_gabor(imagem, mapa_angulos, mapa_frequencia, kx=0.65, ky=0.65):

    angulo_inclinacao = 3
    imagem = np.double(imagem)
    linhas, colunas = imagem.shape
    imagem_retorno = np.zeros((linhas, colunas))

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq_1d = mapa_frequencia.flatten()
    indice_frequencia = np.array(np.where(freq_1d > 0))
    elementos_n_nulo_frequencia = freq_1d[indice_frequencia]
    elementos_n_nulo_frequencia = np.double(np.round((elementos_n_nulo_frequencia * 100))) / 100
    frequencias_unicas = np.unique(elementos_n_nulo_frequencia)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angulo_inclinacao' increments.
    sigma_x = 1 / frequencias_unicas * kx
    sigma_y = 1 / frequencias_unicas * ky
    tamanho_bloco = int(np.round(3 * np.max([sigma_x, sigma_y])))
    array = np.linspace(-tamanho_bloco, tamanho_bloco, (2 * tamanho_bloco + 1))
    x, y = np.meshgrid(array, array)

    # gabor filter equation
    filtro_referencia = np.exp(-(((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y)))) * \
                np.cos(2 * np.pi * frequencias_unicas[0] * x)
    linhas_filtradas, colunas_filtradas = filtro_referencia.shape
    filtro_gabor = np.array(np.zeros((180 // angulo_inclinacao, linhas_filtradas, colunas_filtradas)))

    # Generate rotated versions of the filter.
    for angulo in range(0, 180 // angulo_inclinacao):
        filtro_rotacao = scipy.ndimage.rotate(filtro_referencia, -(angulo * angulo_inclinacao + 90), reshape=False)
        filtro_gabor[angulo] = filtro_rotacao

    # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angulo_inclinacao)
    indice_max_inclinacao = np.round(180 / angulo_inclinacao)
    indice_inclinacao = np.round(mapa_angulos / np.pi * 180 / angulo_inclinacao)
    for lin in range(0, linhas // 16):
        for col in range(0, colunas // 16):
            if indice_inclinacao[lin][col] < 1:
                indice_inclinacao[lin][col] = indice_inclinacao[lin][col] + indice_max_inclinacao
            if indice_inclinacao[lin][col] > indice_max_inclinacao:
                indice_inclinacao[lin][col] = indice_inclinacao[lin][col] - indice_max_inclinacao

    # Find indices of matrix points greater than maxsze from the image boundary
    tamanho_bloco = int(tamanho_bloco)
    lin_valida, col_valida = np.where(mapa_frequencia > 0)
    indice_final = \
        np.where((lin_valida > tamanho_bloco) & (lin_valida < linhas - tamanho_bloco) & (col_valida > tamanho_bloco) & (
                    col_valida < colunas - tamanho_bloco))

    limite_colun = []

    for k in range(0, np.shape(indice_final)[1]):
        r = lin_valida[indice_final[0][k]]
        c = col_valida[indice_final[0][k]]
        if tamanho_bloco < c < (colunas - tamanho_bloco):
            limite_colun.append(c)
        img_block = imagem[r - tamanho_bloco:r + tamanho_bloco + 1][:, c - tamanho_bloco:c + tamanho_bloco + 1]
        imagem_retorno[r][c] = np.sum(img_block * filtro_gabor[int(indice_inclinacao[r // 16][c // 16]) - 1])

    linha_ini, linha_fin = lin_valida[indice_final[0][0]], lin_valida[indice_final[0][np.shape(indice_final)[1] - 1]]

    limite_linha = [linha_ini, linha_fin]
    limite_colun = [min(limite_colun), max(limite_colun)]

    imagem_gabor = 255 - np.array((imagem_retorno < 0) * 255).astype(np.uint8)

    return imagem_gabor, limite_linha, limite_colun
