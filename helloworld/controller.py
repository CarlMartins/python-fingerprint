from django.views.decorators.csrf import csrf_exempt
import sqlite3
import cv2
import numpy as np
import multiprocessing

from matplotlib import pyplot as plt

from helloworld.tratamentoImagem.extraiMinutias import extraiMinutias


@csrf_exempt
def logar(request):
    bytesDigitalLogin = np.asarray(bytearray(request.FILES.get('imgDigital').read()), dtype=np.uint8)
    digitalLogin = cv2.imdecode(bytesDigitalLogin, 0)  # cv2.IMREAD_UNCHANGED

    minutiasLogin, descriptorLogin, img_minutias_login = extraiMinutias(digitalLogin)

    try:
        # Conectar ao banco
        conexaoSQLite = sqlite3.connect('db.sqlite3')
        cursor = conexaoSQLite.cursor()
        print("Conectado com o banco SQLite")

        querySelect = "SELECT * from TB_Cadastro"
        cursor.execute(querySelect)
        resultados = cursor.fetchall()

        if comparaDigitais(img_minutias_login, descriptorLogin, resultados):
            return None


    except sqlite3.Error as e:
        print(e)

    return None


def comparaDigitais(img_minutias_login, descriptorLogin, resultados):
    threads = []
    for linha in resultados:
        idCadastro = linha[0]
        imgDigital = linha[1]
        nmCadastro = linha[2]
        nvCadastro = linha[3]

        bytesDigitalBanco = np.frombuffer(imgDigital, dtype='uint8')

        # decode the array into an image
        digitalBanco = cv2.imdecode(bytesDigitalBanco, 0)

        t = multiprocessing.Process(target=comparar, args=(img_minutias_login, descriptorLogin, digitalBanco))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    return None


def comparar(img_minutias_login, descriptor_login, digital_banco):
    minutias_banco, descriptor_banco, img_minutias_banco = extraiMinutias(digital_banco)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptor_login, descriptor_banco), key=lambda match: match.distance)

    # Plot keypoints
    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_minutias_login)
    axarr[1].imshow(img_minutias_banco)
    plt.show()

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
