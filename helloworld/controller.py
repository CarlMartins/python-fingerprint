from django.views.decorators.csrf import csrf_exempt
import sqlite3
import cv2
import numpy as np
import threading
import multiprocessing

import helloworld.controller
from helloworld.tratamentoImagem.extraiMinutias import extraiMinutias




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

        if comparaDigitais(minutiasLogin, descriptorLogin,resultados) == True:
            return None


    except sqlite3.Error as e:
        print(e)

    return None


def comparaDigitais(minutiasLogin,descriptorLogin,resultados):

    threads = []
    for linha in resultados:
        idCadastro = linha[0]
        imgDigital = linha[1]
        nmCadastro = linha[2]
        nvCadastro = linha[3]

        #Algorítmo de comparação
        bytesDigitalBanco = np.frombuffer(imgDigital, dtype='uint8')

        # decode the array into an image
        digitalBanco = cv2.imdecode(bytesDigitalBanco, 0)

        # thread = threading.Thread(target=helloworld.controller.comparar,args=(minutiasLogin,descriptorLogin,digitalBanco))
        # threads.append(thread)
        comparar(digitalLogin,minutiasLogin,descriptorLogin,digitalBanco)

        t = multiprocessing.Process(target=comparar, args=(minutiasLogin, descriptorLogin, digitalBanco))
        t.start()
        threads.append(t)

        # threads.append(helloworld.controller.comparar)
        # comparar(minutiasLogin,descriptorLogin,digitalBanco)

    for thread in threads:
        thread.join()

        #digitalBanco => digitalBanco já lida pelo openCV
        #digitalLogin => digitalLogin já lida pelo openCV

    return None


def comparar(minutiasLogin,descriptorLogin,digitalBanco):

    minutiasBanco, descriptorBanco = extraiMinutias(digitalBanco)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptorLogin, descriptorBanco), key=lambda match: match.distance)

    # Plot keypoints
    # img4 = cv2.drawKeypoints(digitalLogin, minutiasLogin, outImage=None)
    # img5 = cv2.drawKeypoints(digitalBanco, minutiasBanco, outImage=None)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img4)
    # axarr[1].imshow(img5)
    # plt.show()
    # # Plot matches
    # img3 = cv2.drawMatches(digitalLogin, minutiasLogin, digitalBanco, minutiasBanco, matches, flags=2, outImg=None)
    # plt.imshow(img3)
    # plt.show()


    # Calculate score
    score = 0;
    for match in matches:
        score += match.distance
    score_threshold = 40
    print(score / len(matches))
    if score / len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")

