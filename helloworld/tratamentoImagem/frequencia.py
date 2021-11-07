import numpy as np
import math
import scipy.ndimage


def frequest(imagem, orientim, tamanho_kernel, comprimento_min_onda, comprimento_max_onda):
    linhas, colunas = np.shape(imagem)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled mapa_angulos before reconstructing the angle again.
    cosseno_angulo = np.cos(2 * orientim)  # np.mean(np.cos(2*orientim))
    seno_angulo = np.sin(2 * orientim)  # np.mean(np.sin(2*orientim))
    angulo_bloco = math.atan2(seno_angulo, cosseno_angulo) / 2

    # Rotate the image block so that the ridges are vertical
    imagem_rotacionada = scipy.ndimage.rotate(imagem, angulo_bloco / np.pi * 180 + 90, axes=(1, 0), reshape=False,
                                              order=3,
                                              mode='nearest')

    # Now crop the image so that the rotated image does not contain any invalid regions.
    tamanho_corte = int(np.fix(linhas / np.sqrt(2)))
    offset = int(np.fix((linhas - tamanho_corte) / 2))
    imagem_rotacionada = imagem_rotacionada[offset:offset + tamanho_corte][:, offset:offset + tamanho_corte]

    # Sum down the columns to get a projection of the grey values down the ridges.
    soma_cistas = np.sum(imagem_rotacionada, axis=0)
    dilatacao = scipy.ndimage.grey_dilation(soma_cistas, tamanho_kernel, structure=np.ones(tamanho_kernel))
    ruido_crista = np.abs(dilatacao - soma_cistas);
    pico_threshold = 2;
    maximo_pontos = (ruido_crista < pico_threshold) & (soma_cistas > np.mean(soma_cistas))
    indice_maximo = np.where(maximo_pontos)
    _, numero_picos = np.shape(indice_maximo)

    # Determine the spatial frequency of the ridges by dividing the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds, the frequency image is set to 0
    if numero_picos < 2:
        bloco_frequencia = np.zeros(imagem.shape)
    else:
        wave_length = (indice_maximo[0][-1] - indice_maximo[0][0]) / (numero_picos - 1)
        if comprimento_min_onda <= wave_length <= comprimento_max_onda:
            bloco_frequencia = 1 / np.double(wave_length) * np.ones(imagem.shape)
        else:
            bloco_frequencia = np.zeros(imagem.shape)
    return bloco_frequencia


def calcular_frequencia(imagem, mascara, orient, tamanho_bloco, tamanho_kernel, comprimento_min_onda,
                        comprimento_max_onda):
    # Function to estimate the fingerprint ridge frequency across a
    # fingerprint image.
    linhas, colunas = imagem.shape
    frequencia = np.zeros((linhas, colunas))

    for linha in range(0, linhas - tamanho_bloco, tamanho_bloco):
        for coluna in range(0, colunas - tamanho_bloco, tamanho_bloco):
            bloco_imagem = imagem[linha:linha + tamanho_bloco][:, coluna:coluna + tamanho_bloco]
            angulo_bloco = orient[linha // tamanho_bloco][coluna // tamanho_bloco]
            if angulo_bloco:
                frequencia[linha:linha + tamanho_bloco][:, coluna:coluna + tamanho_bloco] = frequest(bloco_imagem,
                                                                                                     angulo_bloco,
                                                                                                     tamanho_kernel,
                                                                                                     comprimento_min_onda,
                                                                                                     comprimento_max_onda)

    frequencia = frequencia * mascara
    freq_1d = np.reshape(frequencia, (1, linhas * colunas))
    ind = np.where(freq_1d > 0)
    ind = np.array(ind)
    ind = ind[1, :]
    elementos_n_nulo_frequencia = freq_1d[0][ind]
    frequencia_mediana = np.median(elementos_n_nulo_frequencia) * mascara

    return frequencia_mediana
