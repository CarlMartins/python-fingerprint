import sqlite3
import cv2
import numpy as np

def salvarArquivo(dados,caminho):
    with open(caminho,'wb') as file:
        file.write(dados)

def lerBlob(idCadastro):
    try:
        # Conectar ao banco
        conexaoSQLite = sqlite3.connect('db.sqlite3')
        cursor = conexaoSQLite.cursor()
        print("Conectado com o banco SQLite")

        querySelect = "SELECT * from TB_Cadastro where ID_Cadastro = ?"
        cursor.execute(querySelect,(idCadastro,))
        resultados = cursor.fetchall()
        for linha in resultados:
            print("ID = ",linha[0], "Nome = ",linha[2])
            nome = linha[2]
            imagem = linha[1]

            # use numpy to construct an array from the bytes
            x = np.frombuffer(imagem, dtype='uint8')

            # decode the array into an image
            img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

            # show it
            cv2.imshow("Image Window", img)
            cv2.waitKey(0)



            caminho = "C:\\Users\\yuryr\\Desktop\\" + nome + ".tif"



            salvarArquivo(imagem,caminho)

        cursor.close()
    except sqlite3.Error as e:
        print ("Deu ruim")

    finally:
        if conexaoSQLite:
            conexaoSQLite.close()

lerBlob(3)