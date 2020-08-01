#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:53:11 2020

@author: katiau
"""

## Importando a rapaziada

import os
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt

import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from add_dataset import add_labels, decode_image, normaliza_imagens
from testa_estados import carrega_gabarito, predict_estado


####################################
##### Passo 1 - Extrair dados #####
##################################

TRAIN_DIR = 'dataset_1000/'
TEST_DIR = 'dataset_50/'

# Peguei 2000 imagens da pasta de 1000 para treino
# e 1000 da pasta de 50 para teste

train_image_file_names = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)][0:2000] 
test_image_file_names = [TEST_DIR+i for i in os.listdir(TEST_DIR)][0:1000]

# salvando as imagens de treino e teste em arrays numpy
train_images = decode_image(train_image_file_names)
test_images = decode_image(test_image_file_names)
all_images = train_images + test_images

# adiciona o output, que é o tensor labels
labels = add_labels(train_image_file_names)

# normaliza as imagens, divide cada pixel por 255
test_images = normaliza_imagens(test_images)
train_images = normaliza_imagens(train_images)

# de volta para arrays numpy
train_images = np.asarray(train_images,dtype=np.uint8)
labels = np.asarray(labels).astype('uint8')


##########################################
##### Passo 2 - Configurar o modelo #####
########################################

model = Sequential()

# primeira camada convolucional
model.add(Conv2D(64 , (3,3) , input_shape = (50,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

# segunda camada
model.add(Conv2D(64 , (3,3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

# camada 'dense'
model.add(Flatten())
model.add(Dense(64))

# camada de saida
model.add(Dense(27))
model.add(Activation('sigmoid'))


# configurando o modelo
model.compile(loss = 'binary_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])

# ajustando os pesos do modelo
model.fit(train_images, labels, batch_size = 32, epochs = 3, validation_split = 0.1)

# salvando o modelo
model.save('EstadosBR_CNN.model')

######################################
##### Passo 3 - testar o modelo #####
####################################

## carregando as imagens que iremos usar de teste
img_estados = carrega_gabarito()

from random import randint
num = randint(0,26)
estado = predict_estado(img_estados[num:num+1],model)