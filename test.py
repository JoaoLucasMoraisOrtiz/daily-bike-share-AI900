""" from helpers.help import openCsv
import matplotlib.pyplot as plt

#pega os dados do dataset
data = openCsv('data/daily-bike-share.csv')
data.pop(0)
plot = []

for item in data:
    plot.append(item[-1])

print(plot)

# Plota os dados
plt.plot(plot)

# Mostra o gráfico
plt.show() """

import torch
import torch.nn.functional as F

# Converte a lista para um tensor do PyTorch
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

# Redimensiona o tensor para uma dimensão
tensor = tensor.view(-1)

# Normaliza o tensor
tensor_normalizado = F.normalize(tensor, p=2, dim=0)

# Converte o tensor normalizado de volta para uma lista
lista_normalizada = tensor_normalizado.tolist()

print(lista_normalizada)