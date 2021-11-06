from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
import sqlite3
import cv2
import numpy as np
from multiprocessing import Process, Pipe

from django.http import HttpResponse

from matplotlib import pyplot as plt

from helloworld.tratamentoImagem.extraiMinutias import extrai_minutias


@csrf_exempt
def logar(request):
    print(request.FILES.get('imgDigital').name)
    bytes_digital_login = np.asarray(bytearray(request.FILES.get('imgDigital').read()), dtype=np.uint8)
    digital_login = cv2.imdecode(bytes_digital_login, 0)  # cv2.IMREAD_UNCHANGED

    minutias_login, descriptor_login, img_minutias_login = extrai_minutias(digital_login)

    try:
        # Conectar ao banco
        conexao_sqlite = sqlite3.connect('db.sqlite3')
        cursor = conexao_sqlite.cursor()
        print("Conectado com o banco SQLite")

        query_select = "SELECT * from TB_Cadastro"
        cursor.execute(query_select)
        resultados = cursor.fetchall()

        user = compara_digitais(img_minutias_login, descriptor_login, resultados)
        if user is not None:
            print(user, ' logou')
            # login(request, user)
            return HttpResponse("BEM VINDO")
        else:
            return HttpResponse("NAO LOGOU")

    except sqlite3.Error as e:
        print(e)


def compara_digitais(img_minutias_login, descriptor_login, resultados):
    threads = []
    pipes = []
    for linha in resultados:
        id_cadastro = linha[0]
        img_digital = linha[1]
        nm_cadastro = linha[2]
        nv_cadastro = linha[3]

        bytes_digital_banco = np.frombuffer(img_digital, dtype='uint8')

        # decode the array into an image
        digital_banco = cv2.imdecode(bytes_digital_banco, 0)

        pipe_pai, pipe_filho = Pipe()
        returns = []

        t = Process(target=comparar,
                    args=(img_minutias_login, descriptor_login, id_cadastro, nm_cadastro, nv_cadastro, digital_banco,
                          pipe_filho))
        t.start()
        pipes.append(pipe_pai)
        threads.append(t)

    for thread in threads:
        thread.join()

    user = None

    for pipe in pipes:
        retorno = pipe.recv()
        if retorno:
            if user is None:
                user = retorno
            else:
                break

    return user


def comparar(img_minutias_login, descriptor_login, id_cadastro, nm_cadastro, nv_cadastro, digital_banco, pipe_filho):
    minutias_banco, descriptor_banco, img_minutias_banco = extrai_minutias(digital_banco)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptor_login, descriptor_banco), key=lambda match: match.distance)

    # Calculate score
    score = 0;
    for match in matches:
        score += match.distance
    score_threshold = 50
    # print(score / len(matches))
    if score / len(matches) <= score_threshold:
        print("Digital compatível / ", (score / len(matches)))
        pipe_filho.send([id_cadastro, nm_cadastro, nv_cadastro])
        pipe_filho.close()

        # Plot keypoints
        _, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img_minutias_login)
        axarr[1].imshow(img_minutias_banco)
        plt.show()
    else:
        print("Digital incompatível. / ", (score / len(matches)))
        pipe_filho.send(False)
        pipe_filho.close()
