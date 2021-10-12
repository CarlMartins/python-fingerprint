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

        if comparaDigitais(digitalLogin, resultados) == True:
            return None


    except sqlite3.Error as e:
        print(e)

    return None


def comparaDigitais(digitalLogin,resultados):
    for linha in resultados:
        idCadastro = linha[0]
        digitalBanco = linha[1]
        nmCadastro = linha[2]
        nvCadastro = linha[3]

        #Algorítmo de comparação

    return None
