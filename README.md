# CNN para Estados Brasileiros

## Descrição

Já pensou em ensinar geografia para seu computador? 

Com este programa, você pode!

Aqui temos uma Rede Neural Convolucional (cnn) criada no [tensorflow](https://www.tensorflow.org/?hl=pt-br) para reconhecimento das bandeiras dos estados brasileiros.


## Requisitos

- TensorFlow
- OpenCV

## Funcionamento

O código é separado em 3 partes: 

- Pré-processamento: carrega-se as imagens de uma pasta, o que cria o nosso dataset, e codificamos cada estado em um vetor com 27 componentes, a saída da nossa rede.
O Dataset usado contém 27000 imagens e está disponível [neste link da kaggle](https://www.kaggle.com/relampagomarquinhos/flags-of-brazilian-states).
 
- Criar e configurar o modelo: os parâmetros da rede convolucional são adicionados e a rede é treinada.

- Avaliar o modelo: A função predict_estado faz essa brincaceira, apresenta como saída a imagem de input seguido pelo estado que a nossa rede previu.
