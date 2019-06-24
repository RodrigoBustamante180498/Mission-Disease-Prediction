import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report 
import pickle

#Carregamento dos dados
dataset = pandas.read_csv("Mission_Prediction_Dataset.csv")
matriz = dataset.values
X = matriz[:,0:13]
Y = matriz[:,13]

#Carregamento do modelo gerado pelo script de treinamento
modelo = pickle.load(open('MLP1.sav', 'rb'))

#Definição da escala para os dados carregados
scaler = StandardScaler()
scaler.fit(X)

#Criação da lista em que os inputs fornecidos pelo usuário serão armazenados
vetorDeTeste = []

#Solicitação dos dados de input para o usuário
print("Insira o numero de vetores de teste que serão usados como input:")
numeroDeVetores = int(input())
for i in range(numeroDeVetores):
    print("\nInsira os inputs do vetor " + str(i) + ":")
    teste = [float(i) for i in input().split(",")]
    vetorDeTeste.append(teste)

#Tratamento dos dados inputados pelo usuário para adequá-los à escala dos dados carregados
vetorDeTeste = scaler.transform(vetorDeTeste)

#Resultados da previsão do modelo para os dados de input fornecidos pelo usuário e report da porcentagem de acertos
resultados = modelo.predict(vetorDeTeste)
print("\nOutput do modelo:")
print(resultados)