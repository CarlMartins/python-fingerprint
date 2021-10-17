from tratamento import * #importa toda as classes do documento de tratamento
import os
import cv2 as cv #lembrar de tirar / apenas para testes -----------------


#caminho para a pasta do arquivo python
caminhoAtual = os.path.dirname(os.path.abspath(__file__))

#imagem na pasta
img = cv.imread(caminhoAtual + '\database\\1_2.tif')

img = Tratamento.saida(img)


cv.imshow('imagem', img)
cv.waitKey()
