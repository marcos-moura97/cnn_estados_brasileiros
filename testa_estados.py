#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:53:16 2020

@author: katiau
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from add_dataset import normaliza_imagens

def carrega_gabarito():

    nome_estado = ['ac','al','ap','am',
                   'ba','ce','es','go',
                   'ma','mt','ms','mg',
                   'pa','pb','pe','pr',
                   'pi','rj','rn','rs',
                   'rr','ro','sc','sp',
                   'se','to','df']
    
    img_estado = []
    
    for i in range(len(nome_estado)):
        
        print('gabarito/'+nome_estado[i]+'.png')
        img = cv2.imread('gabarito/'+nome_estado[i]+'.png')
        img_estado.append(cv2.resize(img,(100,50)))
        
    return img_estado


def predict_estado(imagem,model):
    plt.imshow(imagem[0])
    imagem = normaliza_imagens(imagem)
    imagem = np.asarray(imagem,dtype=np.uint8)
    prediction = model.predict(imagem)
    #prediction = np.round(prediction)
    print(prediction)
    estado = ''
    ## vendo qual e o estado
    if prediction[0][0] == prediction.max():
        estado = 'Acre'
        
    elif prediction[0][1] == prediction.max():
        estado = 'Alagoas'
        
    elif prediction[0][2] == prediction.max():
        estado = 'Amapa'
        
    elif prediction[0][3] == prediction.max():
        estado = 'Amazonas'
        
    elif prediction[0][4] == prediction.max():
        estado = 'Bahia'
        
    elif prediction[0][5] == prediction.max():
        estado = 'Ceara'
        
    elif prediction[0][6] == prediction.max():
        estado = 'Espirito Santo'
        
    elif prediction[0][7] == prediction.max():
        estado = 'Goias'
        
    elif prediction[0][8] == prediction.max():
        estado = 'Maranhao'
        
    elif prediction[0][9] == prediction.max():
        estado = 'Mato Grosso'
        
    elif prediction[0][10] == prediction.max():
        estado = 'Mato Grosso do Sul'
        
    elif prediction[0][11] == prediction.max():
        estado = 'Minas Gerais'
        
    elif prediction[0][12] == prediction.max():
        estado = 'Para'
        
    elif prediction[0][13] == prediction.max():
        estado = 'Paraiba'
        
    elif prediction[0][14] == prediction.max():
        estado = 'Pernambuco'
        
    elif prediction[0][15] == prediction.max():
        estado = 'Parana'
        
    elif prediction[0][16] == prediction.max():
        estado = 'Piaui'
        
    elif prediction[0][17] == prediction.max():
        estado = 'Rio de Janeiro'
        
    elif prediction[0][18] == prediction.max():
        estado = 'Rio Grande do Norte'
        
    elif prediction[0][19] == prediction.max():
        estado = 'Rio Grande do Sul'
        
    elif prediction[0][20] == prediction.max():
        estado = 'Roraima'
        
    elif prediction[0][21] == prediction.max():
        estado = 'Rondonia'
        
    elif prediction[0][22] == prediction.max():
        estado = 'Santa Catarina'
        
    elif prediction[0][23] == prediction.max():
        estado = 'Sao Paulo'
        
    elif prediction[0][24] == prediction.max():
        estado = 'Sergipe'
        
    elif prediction[0][25] == prediction.max():
        estado = 'Tocantins'
        
    elif prediction[0][26] == prediction.max():
        estado = 'Distrito Federal'
        
    else:
        print('Deu ruim')
    
    print("##################################")
    print('Bandeira do estado do: ',estado)
    print("##################################")

    return estado
        
        
