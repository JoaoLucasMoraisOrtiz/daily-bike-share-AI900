import csv
from random import shuffle
import torch

def openCsv(dir):
    """ 
        Abre um arquivo csv e retorna uma lista com os dados do arquivo
        @param dir: string - Diretório do arquivo csv
        @return data: list - Lista com os dados do arquivo
    """
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def handleData(data):
    """ 
        Trata os dados do dataset
        @param data: list - Lista com os dados do dataset
        @return trainData: list - Lista com os dados de treino
        @return validationData: list - Lista com os dados de validação
        @return header: list - Lista com o cabeçalho do dataset
    """

    #salva o cabeçalho
    header = data.pop(0)

    #embaralha o dataset
    shuffle(data)

    #separa os dados de treino e de validação
    trainData = data[:int(len(data)*0.75)].copy()
    validationData = data[int(len(data)*0.75):].copy()

    for i in range(len(trainData)):
        trainData[i] = list(map(float, trainData[i]))

    for i in range(len(validationData)):
        validationData[i] = list(map(float, validationData[i]))

    return torch.tensor(trainData), torch.tensor(validationData), header

def normalize(vetor):
    """ 
        Normaliza um vetor
        @param vetor: tensor - Vetor a ser normalizado
        @return vetor_normalizado: tensor - Vetor normalizado
        @return media: tensor - Média do vetor
        @return desvio_padrao: tensor - Desvio padrão do vetor
    """
    media = torch.mean(vetor)
    desvio_padrao = torch.std(vetor)
    vetor_normalizado = (vetor - media) / desvio_padrao
    return vetor_normalizado, media, desvio_padrao

def denormalize(vetor_normalizado, media, desvio_padrao):
    """ 
        Desnormaliza um vetor
        @param vetor_normalizado: tensor - Vetor normalizado
        @param media: tensor - Média do vetor
        @param desvio_padrao: tensor - Desvio padrão do vetor
        @return vetor_original: tensor - Vetor desnormalizado
    """
    vetor_original = vetor_normalizado * desvio_padrao + media
    return vetor_original   