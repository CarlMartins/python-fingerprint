import cv2

from helloworld.tratamentoImagem.normalizacao import normalizar
from helloworld.tratamentoImagem.segmentacao import segmentar_imagem
from helloworld.tratamentoImagem.calculoAngulos import *
from helloworld.tratamentoImagem.frequencia import calcular_frequencia
from helloworld.tratamentoImagem.filtroGabor import filtro_gabor
from helloworld.tratamentoImagem.esqueletizacao import afinar_digital
from helloworld.tratamentoImagem.crossing_number import identificar_minucias
from helloworld.tratamentoImagem.poincare import calcular_singularidades


def extrai_minutias(imagem):

    tamanho_bloco = 16

    imagem_normalizada = normalizar(imagem.copy(), float(150), float(150))

    (imagem_segmentada, img_seg_normalizada, mascara) = segmentar_imagem(imagem_normalizada, tamanho_bloco, 0.4)

    imagem_normalizada = cv2.blur(imagem_normalizada, (3, 3))

    mapa_angulos = calcular_angulos(imagem_normalizada, tamanho_bloco)

    mapa_frequencia = calcular_frequencia(img_seg_normalizada, mascara, mapa_angulos, tamanho_bloco,
                                          tamanho_bloco, minWaveLength=10, maxWaveLength=15)

    imagem_gabor, limite_linha, limite_coluna = filtro_gabor(img_seg_normalizada, mapa_angulos, mapa_frequencia)

    imagem_digital_afinada = afinar_digital(imagem_gabor)

    raw_minutias, minucias, coordenadas_minucias = identificar_minucias(imagem, imagem_digital_afinada, mapa_frequencia,
                                                                        limite_linha, limite_coluna, 3)

    orb = cv2.ORB_create()
    
    # Compute descriptors
    _, descriptor = orb.compute(imagem_digital_afinada, coordenadas_minucias)  # Retornar desLogin
    
    # singularidades = calcular_singularidades(imagem_digital_afinada, mapa_angulos, 1, tamanho_bloco, mascara)
    # imagem_mapa_angulos = gerar_imagem_angulos(imagem_segmentada, mascara, mapa_angulos, tamanho_bloco=tamanho_bloco)
    # _, threshold_im = cv2.threshold(imagem_normalizada, 127, 255, cv2.THRESH_OTSU)
    
    # cv2.imshow("teste", imagem_normalizada)
    # cv2.imshow("teste2", threshold_im)
    # cv2.imshow("teste3", imagem_segmentada)
    # cv2.imshow("teste4", img_seg_normalizada)
    # cv2.imshow("teste5", imagem_mapa_angulos)
    # cv2.imshow("teste6", mapa_frequencia)
    # cv2.imshow("teste7", imagem_gabor)
    # cv2.imshow("teste8", imagem_digital_afinada)
    # cv2.imshow("teste9", minucias)
    # cv2.imshow("teste10", singularidades)
    # cv2.imshow("teste11", raw_minutias)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return coordenadas_minucias, descriptor,raw_minutias
