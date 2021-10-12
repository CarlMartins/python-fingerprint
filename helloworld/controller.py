from django.views.decorators.csrf import csrf_exempt
import sqlite3
import cv2
import numpy as np

@csrf_exempt
def logar(request):
    digitalLogin = request.POST.get('imgDigital')

    try:
        # Conectar ao banco
        conexaoSQLite = sqlite3.connect('db.sqlite3')
        cursor = conexaoSQLite.cursor()
        print("Conectado com o banco SQLite")

        querySelect = "SELECT * from TB_Cadastro"
        cursor.execute(querySelect)
        resultados = cursor.fetchall()

        for linha in resultados:
            idCadastro = linha[0]
            digitalBanco = linha[1]
            nmCadastro = linha[2]
            nvCadastro = linha[3]

            if comparaDigitais(digitalLogin,digitalBanco) == True:
                return "foi"

    except Exception as e:
        print("deu ruim")

    return None


def comparaDigitais(imgDigital):
    return None
