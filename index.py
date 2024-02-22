from helpers.help import openCsv, handleData, normalize, denormalize
from network import NeuralNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#pega os dados do dataset
data = openCsv('data/daily-bike-share.csv')

#trainDataNotNormalize, validationDataNotNormalize, header = handleData(data)
trainDataNotNormalize, validationDataNotNormalize, header = handleData(data)
del data

#normaliza os dados
trainData = [0 for _ in range(len(trainDataNotNormalize))]
mediaTrain = [0 for _ in range(len(trainDataNotNormalize))]
desvPadraoTrain = [0 for _ in range(len(trainDataNotNormalize))]
for item in range(0, len(trainDataNotNormalize)):
    trainData[item], mediaTrain[item], desvPadraoTrain[item] = normalize(trainDataNotNormalize[item])

validationData = [0 for _ in range(len(validationDataNotNormalize))]
media = [0 for _ in range(len(validationDataNotNormalize))]
desvPadrao = [0 for _ in range(len(validationDataNotNormalize))]
for item in range(0, len(validationDataNotNormalize)):
    validationData[item], media[item], desvPadrao[item] = normalize(validationDataNotNormalize[item])

#cria a rede neural
net = NeuralNetwork.Net(12, [48, 96, 112, 224], 1)

#dados para o terinamento
gradienteErro = nn.MSELoss()
#equivale a nossa função de correção de erro, onde usavamos a regra delta, agora vamos utilizar um optimizer
correcaoErro = optim.SGD(net.parameters(), lr=0.01, momentum=0.7)

eraError = torch.tensor([0.0], dtype=torch.float64)
lr = float()

#por 400 epocas
for i in range(400):
    """ if i%20==0:
        lr = 10.01 - (i/40)
        if lr < 0.01:
            lr = 0.01
        correcaoErro = optim.Adam(net.parameters(), lr=lr) """

    print(round(i/40)*'#'+f' {i/4}%')
    
    for item in trainData:

        #zera o gradiente
        correcaoErro.zero_grad()

        #faz a passagem para frente
        saida = net(item[:-1])
        #calcula o erro
        erro = gradienteErro(saida[0], item[-1])
        eraError += erro

        #faz a propagração do erro para trás
        erro.backward()

        #corrige os pesos
        correcaoErro.step()

    print(f'erro: {eraError/len(trainData)}')
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    eraError = 0
# Cria uma lista com os resultados esperados
resultados_esperados = [item[-1] for item in trainData]

esperado = []
resps = []
for item in validationData:
    #faz a passagem para frente
    saida = net(item[:-1])
    
    #print(f'saida da rede: {saida} /-/ resultado esperado: {item[-1]}')
    esperado.append(item[-1])
    resps.append(saida[0])

print('-=-=-=-=-=-=-=-=-=-=-=-=-')


for item in esperado:
    print(f'esperado: {item}')

for item in resps:
    print(f'resposta: {item}')

esperadoDenormalize = []
respsDenormalize = []

for item in range(0, len(esperado)):
    esperadoDenormalize.append(denormalize(esperado[item], media[item], desvPadrao[item]))
plt.plot(esperadoDenormalize, color='blue')


for item in range(0, len(resps)):
    respsDenormalize.append(denormalize(resps[item], media[item], desvPadrao[item]).detach().numpy())

absError = []
for i in range(len(esperadoDenormalize)):
    absError.append(abs(esperadoDenormalize[i] - respsDenormalize[i]))
    print(f'esperado: {esperadoDenormalize[i]} /-/ resps: {respsDenormalize[i]}')


print( f'erro médio absoluto: {sum(absError) / float(len(absError))}' )
plt.plot(respsDenormalize, color='red')
plt.figtext(0.15, 0.85, f'Erro médio absoluto: {sum(absError) / float(len(absError))}', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
# Mostra o gráfico
plt.show()


print('-=-=-=-=-=-=-=-=-=-=-=-=-')