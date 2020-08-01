#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf


# Funcao que lê as imagens das listas de nomes e salva em uma na forma de array
# numpy
def decode_image(image_file_names, resize_func=None):
    
    images = []
    
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()   
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i+1) % 1000 == 0:
                print('Imagens processadas: ',i+1)
        
        session.close()
    return images

# esta funcao gera a saida, um array de 27 elementos em que cada um representa
# um estado
def add_labels(nome_imagens):
    #estado = np.zeros(27)
    labels = []
    
    for nome in nome_imagens:
        if 'ac' in nome:
            ac = np.zeros(27)
            ac[0] = 1
            labels.append(ac);
            
        elif 'al' in nome:
            al = np.zeros(27)
            al[1] = 1
            labels.append(al);
            
        elif 'ap' in nome:
            ap = np.zeros(27)
            ap[2] = 1
            labels.append(ap);
            
        elif 'am' in nome:
            am = np.zeros(27)
            am[3] = 1
            labels.append(am);
            
        elif 'ba' in nome:
            ba = np.zeros(27)
            ba[4] = 1
            labels.append(ba);
            
        elif 'ce' in nome:
            ce = np.zeros(27)
            ce[5] = 1
            labels.append(ce);
            
        elif 'es' in nome:
            es = np.zeros(27)
            es[6] = 1
            labels.append(es);
            
        elif 'go' in nome:
            go = np.zeros(27)
            go[7] = 1
            labels.append(go);
            
        elif 'ma' in nome:
            ma = np.zeros(27)
            ma[8] = 1
            labels.append(ma);
            
        elif 'mt' in nome:
            mt = np.zeros(27)
            mt[9] = 1
            labels.append(mt);
        
        elif 'ms' in nome:
            ms = np.zeros(27)
            ms[10] = 1
            labels.append(ms);
            
        elif 'mg' in nome:
            mg = np.zeros(27)
            mg[11] = 1
            labels.append(mg);
            
        elif 'pa' in nome:
            pa = np.zeros(27)
            pa[12] = 1
            labels.append(pa);
            
        elif 'pb' in nome:
            pb = np.zeros(27)
            pb[13] = 1
            labels.append(pb);
            
        elif 'pe' in nome:
            pe = np.zeros(27)
            pe[14] = 1
            labels.append(pe);
            
        elif 'pr' in nome:
            pr = np.zeros(27)
            pr[15] = 1
            labels.append(pr);
            
        elif 'pi' in nome:
            pi = np.zeros(27)
            pi[16] = 1
            labels.append(pi);
            
        elif 'rj' in nome:
            rj = np.zeros(27)
            rj[17] = 1
            labels.append(rj);
            
        elif 'rn' in nome:
            rn = np.zeros(27)
            rn[18] = 1
            labels.append(rn);
            
        elif 'rs' in nome:
            rs = np.zeros(27)
            rs[19] = 1
            labels.append(rs);
            
        elif 'rr' in nome:
            rr = np.zeros(27)
            rr[20] = 1
            labels.append(rr);
            
        elif 'ro' in nome:
            ro = np.zeros(27)
            ro[21] = 1
            labels.append(ro);
            
        elif 'sc' in nome:
            sc = np.zeros(27)
            sc[22] = 1
            labels.append(sc);
            
        elif 'sp' in nome:
            sp = np.zeros(27)
            sp[23] = 1
            labels.append(sp);
            
        elif 'se' in nome:
            se = np.zeros(27)
            se[24] = 1
            labels.append(se);
            
        elif 'to' in nome:
            to = np.zeros(27)
            to[25] = 1
            labels.append(to);
            
        elif 'df' in nome:
            df = np.zeros(27)
            df[26] = 1
            labels.append(df);
            
    return labels
    
# Funcao que normaliza os arrays para teste e treino
def normaliza_imagens(teste):    
    ## normalizando as imagens
    for i in range(len(teste)):
        teste[i] = teste[i].astype(np.uint8)
        teste[i] = teste[i]/255.0

    return teste
