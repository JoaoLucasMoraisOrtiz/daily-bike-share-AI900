import torch
import torch.nn as nn

class Net(torch.nn.Module):
    """ 
        Classe que representa a rede neural
    """
    def __init__(self, input_size, hidden_size, output_size):
        """ 
            Inicializa a rede neural
            @param input_size: int - Tamanho da entrada da rede
            @param hidden_size: list - Lista com o tamanho das camadas escondidas
            @param output_size: int - Tamanho da saída da rede 
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.out = nn.Linear(hidden_size[3], output_size)
    
    def forward(self, x):
        """ 
            Faz a passagem para frente da rede neural
            @param x: tensor - Entrada da rede
            @return x: tensor - Saída da rede
        """
        x = torch.sigmoid(self.fc1(x))
        x = nn.functional.tanh(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = nn.functional.leaky_relu(self.out(x))
        return x