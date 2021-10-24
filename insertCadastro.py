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
        conexaoSQLite = sqlite3.connect('db.sqlite3')
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


insertBlob("Ministro Cesar Xavier",3,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\1_1.png")

insertBlob("Yury",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\14_5.png")
insertBlob("Carlitos",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\2_1.png")
insertBlob("Alysson",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\11_2.png")
insertBlob("Fabrício",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\3_5.png")
insertBlob("Augusto",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\13_3.png")

insertBlob("Funcionário 1",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\16_2.png")
insertBlob("Funcionário 2",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\12_1.png")
insertBlob("Funcionário 3",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\10_2.png")
insertBlob("Funcionário 4",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
                        "Semestre\\Projeto\\python-fingerprint\\fingerprints\\XAVIER\\6_2.png")









# insertBlob("Ministro Cesar Xavier",3,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\101\\101_1.tif")
#
# insertBlob("Yury",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\102\\102_2.tif")
# insertBlob("Carlitos",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\103\\103_1.tif")
# insertBlob("Alysson",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\104\\104_1.tif")
# insertBlob("Fabrício",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\105\\105_7.tif")
# insertBlob("Augusto",2,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\106\\106_6.tif")
#
# insertBlob("Funcionário 1",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\107\\107_7.tif")
# insertBlob("Funcionário 2",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\108\\108_6.tif")
# insertBlob("Funcionário 3",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\109\\109_3.tif")
# insertBlob("Funcionário 4",1,"C:\\Yury\\UNIP\\6º Semestre\\PIVC - Processamento de Imagens e Visão Computacional\\APS 6º "
#                         "Semestre\\Projeto\\python-fingerprint\\fingerprints\\110\\110_3.tif")