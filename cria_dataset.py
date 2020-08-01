#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este arquivo cria o dataset com N figuras por imagem

@author: katiau
"""

import cv2

estados = ['ac','al','ap','am',
           'ba','ce','es','go',
           'ma','mt','ms','mg',
           'pa','pb','pe','pr',
           'pi','rj','rn','rs',
           'rr','ro','sc','sp',
           'se','to','df']


for i in range(len(estados)):
    
    print('gabarito/'+estados[i]+'.png')
    estado_antigo = cv2.imread('dataset/'+estados[i]+'.png')

    estado_novo = cv2.resize(estado_antigo,(100,50))
    
    for j in range(1,1001):
    
        cv2.imwrite('dataset_1000/'+estados[i]+str(j)+'.png',estado_novo)