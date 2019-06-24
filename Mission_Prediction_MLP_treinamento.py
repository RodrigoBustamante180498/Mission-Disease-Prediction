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

#Separaçao entre dados de treino e dados de teste
test_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#Normalização dos dados
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

#Criação do multi layer perceptron e treinamento
mlp = MLPClassifier(hidden_layer_sizes=(20), max_iter=1000, activation = 'relu', solver = 'adam')  
mlp.fit(X_train, Y_train)

#Simulação para os dados de teste
Y_pred = mlp.predict(X_test)

#Report da precisão do modelo para os dados de teste (porcentagem de acertos para outputs esperados iguais a 0 e iguais a 1 separados)
print(classification_report(Y_test,Y_pred))

#Salva o modelo
filename = 'MLP1.sav'
pickle.dump(mlp, open(filename, 'wb'))

#Carregando o modelo gerado e verificando a porcentagem bruta de acerto dele para os dados de teste utilizados (porcentagem geral de acertos para outputs 0 e 1 juntos)
modelo = pickle.load(open(filename, 'rb'))
result = modelo.score(X_test, Y_test)
print("\nAcurácia do modelo para os dados de teste:")
print(result)
resultados = modelo.predict(X_test)
print(classification_report(Y_test,resultados))