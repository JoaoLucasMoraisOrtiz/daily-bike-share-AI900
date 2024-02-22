# Daily Bike Share (AI900)
Este desafio foi proposto no curso AI900 da Microsoft em parseria com a DIO no curso AI-900.

# Tecnologias Utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pytorch](https://img.shields.io/badge/python-gray?style=for-the-badge&logo=pytorch&logoColor=red)

## tree do projeto
| Pasta   | Conteúdo   |
|--------|------------|
|network| Contém toda a nossa rede neural|
|helpers| Contém funções necessárias para lidar com os dados do projeto, ou seja carregar e preparar o dataset|
|graphics| Apresenta os gráficos mostrando uma evolução do projeto com diferentes testes que realizei, até encontrar a melhor [solução](https://github.com/JoaoLucasMoraisOrtiz/daily-bike-share-AI900/blob/main/graphics/sigmoid%2Btanh%2Bsigmoid%2Btanh%2Bleaky_relu(SGD%2BMSE)ShuffleData.png)
|data| Contém nosso dataset|

# Passos do Projeto
Para este projeto comecei criando os helpers para lidar com o arquivo CSV do banco de dados.
Após isto plotei o gráfico com o auxilio do matplotlib, e tentei reconhecer qual padrão seguia o aluguel das bicicletas (que é a métrica que queremos fazer a predição).

Pude observar que ela seguia um modelo com algum padrão, sempre com uma montanha e um vale em seguida.

Tentei utilizar a função seno para emular isto, mas não tive bons resultados. Portanto retornei para as funções mais tradicionais (sigmoide, tang. hiperb., softsign, entre outras).

Por conta da repetição de padrões no gráfico, tentei alterar o otimizador de Adam para SDG e adcionar um termo momentum.

Percebi então que tinha um problema de atingir os picos, mas que já alcançava os vales. Neste momento resulvi alterar a função da última camada para dar mais liberdade para os picos trocando-a para a função ReLu. Isto não funcionou, pois eu comecei a não conseguir alcançar os vales, então troquei para uma diferenciação da função ReLu que permite valores negativos, a leaky ReLu.