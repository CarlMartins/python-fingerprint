from matplotlib import pyplot as plt
import cv2 as cv

import numpy as np


class Tratamento:
    #Estrutura, devolve um array, que vai devolver a natureza da operação https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    KERNEL = cv.getStructuringElement(cv.MORPH_CROSS, (3,3)) #CONST

    #aplica os primeiros filtros na imagem
    def filtros(img):
        #joga um filtro cinza na imagem
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #filtro para tirar imperfeições
        img = cv.blur(img,(2,2))
        #imagem = cv.GaussianBlur(imagem,(5,5),0)

        return img

    #aumenta a gama da imagem
    def ajustaGama(img, gama):
        #mais fácil para trabalhar com a gama
        gama = 1.0 / gama
        array = np.array([((i / 255.0) ** gama) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv.LUT(img, array)

    #binariza a imagem
    #O threshold obrigatóriamente deve ser atribuido a uma tupla ou a duas variáveis distinstas
    def binariza(img):

        #mesmo não seno usado, esse função precisa ser atribuida a duas variáveis
        ret, thresh = cv.threshold(img, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresh

    #transforma a imagem em um "esqueleto"
    def skeletizacao(img):
        skel = np.zeros(img.shape, np.uint8)

        while True:
            #seria o mesmo que "afinar" a imagem,
            erosao = cv.erode(img, Tratamento.KERNEL)

            #Aumenta o traço da imagem,
            dilatacao = cv.dilate(erosao, Tratamento.KERNEL)
            sub = cv.subtract(img, dilatacao)
            skel = cv.bitwise_or(skel, sub)
            img = erosao.copy()

            if cv.countNonZero(img) == 0:
                break

        return skel

    #tentativa de tirar as imperfeições da imagem
    def limpa(img):
        img = cv.morphologyEx(img, cv.MORPH_OPEN, Tratamento.KERNEL)

        return img


    def linhas(img):
        img = cv.Canny(img,50,150,apertureSize = 3)

        return img

    #retorno da imagem
    def saida(img):
        img = Tratamento.filtros(img)
        img = Tratamento.binariza(img)
        #imagem = Tratamento.linhas(imagem) #isto é um problema
        #imagem = Tratamento.limpa(imagem)
        img = Tratamento.ajustaGama(img, 0.5)
        img = Tratamento.skeletizacao(img)

        return img





class Utils:

    def Hogh(img):
        pass

        return img



    def saida(img):
        img = Utils.Hogh(img)


        return img