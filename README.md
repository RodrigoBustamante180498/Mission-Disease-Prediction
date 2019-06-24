# Mission_Prediction_SKL

O projeto contém 2 arquivos de código: Mission_Prediction_MLP_treinamento e MissionPrediction_MLP_teste_usuario. O desafio consiste em ler dados de um arquivo .csv e utilizar as primeiras 13 colunas de cada linha como inputs para um modelo, enquanto a 14ª linha consiste em um output binário.

 Para utilizar os códigos é necessário que o arquivo .csv que contém os dados esteja no diretório do programa.

# Mission_Prediction_MLP_treinamento 
O arquivo Mission_Prediction_MLP_treinamento é um código que carrega os dados do arquivo .csv, monta um modelo MLP (Multi Layer Perceptron) e realiza as etapas de treinamento do mesmo. Nesse código, os dados obtidos são separados de forma aleatória, de modo que 70% deles são utilizados para o treinamento e 30% são utilizados para testar o modelo após o treinamento, fornecendo estatísticas de porcentagem de acertos. 

 

# MissionPrediction_MLP_teste_usuario
O arquivo MissionPrediction_MLP_teste_usuario é um código que pede para que o usuário insira inicialmente quantos dados de teste o usuário irá querer testar (número inteiro). Em seguida, o programa pede para que o usuário insira uma uma linha com os inputs (13) separados por vírgulas, em seguida, insira a próxima linha de inputs até que atinja o número de testes inicialmente informado pelo usuário. Após inserir o número de dados de teste desejados, o programa carrega o modelo salvo pelo script de treinamento e realiza a simulação para esses dados de teste fornecidos pelo usuário, gerando as respostas obtidas pelo modelo para os valores inseridos.
