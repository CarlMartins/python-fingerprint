import cv2

from helloworld.tratamentoImagem.normalizacao import normalize
from helloworld.tratamentoImagem.segmentacao import create_segmented_and_variance_images
from helloworld.tratamentoImagem.calculoAngulos import *
from helloworld.tratamentoImagem.frequencia import ridge_freq
from helloworld.tratamentoImagem.filtroGabor import gabor_filter
from helloworld.tratamentoImagem.esqueletizacao import skeletonize
from helloworld.tratamentoImagem.crossing_number import calculate_minutiaes
from helloworld.tratamentoImagem.poincare import calculate_singularities


def extraiMinutias(imagem):
    block_size = 16
    imagem_normalizada = normalize(imagem.copy(), float(100), float(100))
    _, threshold_im = cv2.threshold(imagem_normalizada, 127, 255, cv2.THRESH_OTSU)
    (segmented_img, normim, mask) = create_segmented_and_variance_images(imagem_normalizada, block_size, 0.4)
    imagem_normalizada = cv2.blur(imagem_normalizada, (3, 3))
    angles = calculate_angles(imagem_normalizada, W=block_size, smoth=False)
    orientation_img = visualize_angles(segmented_img, mask, angles, W=block_size)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=9, minWaveLength=10, maxWaveLength=15)
    gabor_img, limite_linha, limite_colun = gabor_filter(normim, angles, freq)
    skel_img = skeletonize(gabor_img)
    raw_minutias, minutias, coordenadas_minutias = calculate_minutiaes(imagem, skel_img, freq,
                                                                       limite_linha, limite_colun,kernel_size=5)
    orb = cv2.ORB_create()
    # Compute descriptors
    _, descriptor = orb.compute(skel_img, coordenadas_minutias)  # Retornar desLogin
    singularities = calculate_singularities(skel_img, angles, 1, block_size, mask)

    # cv2.imshow("teste", imagem_normalizada)
    # cv2.imshow("teste2", threshold_im)
    # cv2.imshow("teste3", segmented_img)
    # cv2.imshow("teste4", normim)
    # cv2.imshow("teste5", orientation_img)
    # cv2.imshow("teste6", freq)
    # cv2.imshow("teste7", gabor_img)
    # cv2.imshow("teste8", skel_img)
    # cv2.imshow("teste9", minutias)
    # cv2.imshow("teste10", singularities)
    # cv2.imshow("teste11", raw_minutias)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return coordenadas_minutias, descriptor,raw_minutias
