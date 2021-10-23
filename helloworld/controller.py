from django.views.decorators.csrf import csrf_exempt
import sqlite3
import cv2
import numpy as np

from helloworld.tratamentoImagem.normalizacao import normalize
from helloworld.tratamentoImagem.segmentacao import create_segmented_and_variance_images
from helloworld.tratamentoImagem.calculoAngulos import *
from helloworld.tratamentoImagem.frequencia import ridge_freq
from helloworld.tratamentoImagem.filtroGabor import gabor_filter
from helloworld.tratamentoImagem.esqueletizacao import skeletonize
from helloworld.tratamentoImagem.crossing_number import calculate_minutiaes
from helloworld.tratamentoImagem.poincare import calculate_singularities




@csrf_exempt
def logar(request):
    bytesDigitalLogin = np.asarray(bytearray(request.FILES.get('imgDigital').read()), dtype=np.uint8)
    digitalLogin = cv2.imdecode(bytesDigitalLogin, 0) #cv2.IMREAD_UNCHANGED

    minutiasLogin,descriptorLogin = extraiMinutias(digitalLogin)


    try:
        # Conectar ao banco
        conexaoSQLite = sqlite3.connect('db.sqlite3')
        cursor = conexaoSQLite.cursor()
        print("Conectado com o banco SQLite")

        querySelect = "SELECT * from TB_Cadastro"
        cursor.execute(querySelect)
        resultados = cursor.fetchall()

        if comparaDigitais(digitalLogin, resultados) == True:
            return None


    except sqlite3.Error as e:
        print(e)

    return None


def comparaDigitais(digitalLogin,resultados):
    for linha in resultados:
        idCadastro = linha[0]
        imgDigital = linha[1]
        nmCadastro = linha[2]
        nvCadastro = linha[3]

        #Algorítmo de comparação
        bytesDigitalBanco = np.frombuffer(imgDigital, dtype='uint8')

        # decode the array into an image
        digitalBanco = cv2.imdecode(bytesDigitalBanco, cv2.IMREAD_UNCHANGED)

        #digitalBanco => digitalBanco já lida pelo openCV
        #digitalLogin => digitalLogin já lida pelo openCV

    return None


def extraiMinutias(imagem):
    block_size = 16
    imagemNormalizada = normalize(imagem.copy(), float(100), float(100))
    _, threshold_im = cv2.threshold(imagemNormalizada, 127, 255, cv2.THRESH_OTSU)
    (segmented_img, normim, mask) = create_segmented_and_variance_images(imagemNormalizada, block_size, 0.4)
    imagemNormalizada = cv2.blur(imagemNormalizada, (3, 3))
    angles = calculate_angles(imagemNormalizada, W=block_size, smoth=False)
    orientation_img = visualize_angles(segmented_img, mask, angles, W=block_size)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=9, minWaveLength=10, maxWaveLength=15)
    gabor_img = gabor_filter(normim, angles, freq)
    skel_img = skeletonize(gabor_img)
    rawMinutias,minutias, coordenadasMinutias = calculate_minutiaes(imagem,skel_img, freq,kernel_size=5)  # Retornar coordenadasMinutias
    orb = cv2.ORB_create()
    # Compute descriptors
    _, descriptor = orb.compute(skel_img, coordenadasMinutias)  # Retornar desLogin
    print(coordenadasMinutias)
    singularities = calculate_singularities(skel_img, angles, 1, block_size, mask)

    cv2.imshow("teste", imagemNormalizada)
    cv2.imshow("teste2", threshold_im)
    cv2.imshow("teste3", segmented_img)
    cv2.imshow("teste4", normim)
    cv2.imshow("teste5", orientation_img)
    cv2.imshow("teste6", freq)
    cv2.imshow("teste7", gabor_img)
    cv2.imshow("teste8", skel_img)
    cv2.imshow("teste9", minutias)
    cv2.imshow("teste10", singularities)
    cv2.imshow("teste11", rawMinutias)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return coordenadasMinutias,descriptor