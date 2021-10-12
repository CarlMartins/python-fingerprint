import os.path
import sqlite3

def convertToBinary(filePath):
    #Converter Arquivo em binario para o campo BLOB
    with open(filePath,'rb') as file:
        dadosBlob = file.read()
    return dadosBlob

def insertBlob(nmCadastro,nvCadastro,pathDigital):
    try:
        #Conectar ao banco
        conexaoSQLite = sqlite3.connect("C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\db.sqlite3")
        cursor = conexaoSQLite.cursor()
        print("Conectado com o banco SQLite")

        #Preparar Query
        queryInsert = "Insert into TB_Cadastro (Img_Biometria,Nm_Cadastro,Nv_Cadastro) values (?,?,?)"
        imgBiometria = convertToBinary(pathDigital)
        dados_insert = (imgBiometria,nmCadastro,nvCadastro)

        #Executar Query
        cursor.execute(queryInsert,dados_insert)
        conexaoSQLite.commit()
        print("Insert realizado com sucesso")
        cursor.close()

    except sqlite3.Error as e:
        print("Falha ao inserir Dados")

    finally:
        if conexaoSQLite:
            conexaoSQLite.close()
            print("Conexão com banco fechada")

insertBlob("Ministro",3,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\101\\101_1.tif")