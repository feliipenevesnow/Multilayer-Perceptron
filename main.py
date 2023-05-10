import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import time

# Carregar o conjunto de dados em um DataFrame
dataset = pd.read_csv("star.csv")

# Extrair as características e rótulos do conjunto de dados
x = dataset.iloc[:, 0:5]
y = dataset.iloc[:, 6]
cores = dataset.iloc[:, 5]

espectro = y

# Transformo String em numeros
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
cores = label_encoder.fit_transform(cores)

# Criando dicionario para saber a qual é a classe espectral em letras ao inves de numero
classe_espectral = {}
cont = 0
for i in y:
  classe_espectral[i] = espectro[cont]
  cont += 1

# Concateno as cores, depois da mudança para numero, nos dados de entrada
cores = pd.DataFrame(cores, columns=['cores'])
x = pd.concat([x, cores], axis=1)

y = np.array(y)
x = np.array(x)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalizar as entradas do conjunto de treinamento
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Criar o modelo de rede neural
model = Sequential()

# Adicionar as camadas densas à rede neural
model.add(Dense(32, input_dim=6, activation='relu'))
model.add(Dense(16, activation='relu'))
# Adicionando a camada de saída com 7 neurônios e função de ativação "softmax"
model.add(Dense(7, activation='softmax'))

##############################################################################################
# # Compilar o modelo com Termo de momentum
#from keras.optimizers import Adam
#opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

##############################################################################################
# Compilar o modelo com o otimizador Resilient Propagation
#from keras.optimizers import RMSprop
#optimizer = RMSprop(learning_rate=0.01, rho=0.0)
#model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

inicio_treinamento = time.perf_counter()
# Treinar o modelo
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
fim_treinamento = time.perf_counter()

# Avaliar o modelo nos dados de teste
loss, accuracy = model.evaluate(X_test, y_test)

# Imprimir a perda e a acurácia
print("Perda nos dados de teste:", loss)
print("Acurácia nos dados de teste:", accuracy)

inicio_teste = time.perf_counter()
# Obter as previsões do modelo para o conjunto de teste
y_pred = model.predict(X_test)
fim_teste = time.perf_counter()

# Arredondar as previsões para o valor mais próximo (0 ou 1)
y_pred = np.round(y_pred)

# Comparar as previsões com as respostas reais
for i in range(len(y_test)):
  pred = np.argmax(y_pred[i])
  real = y_test[i]
  if pred == real:
      print("Teste ", i+1, ": Previsão =", y_pred[i], " | ", pred, " | ", classe_espectral[pred], ", Resposta real =", real, " | ", classe_espectral[real])
  else:
      print("Teste ", i + 1, ": Previsão =", y_pred[i], " | ", pred, " | ", classe_espectral[pred], ", Resposta real =", real, " | ", classe_espectral[real], " - ERROU")




res = int(input("\nAbrir matriz confusão? (1 - SIM, 0 - NÃO)\n"))




if res == 1:
  # Gerar a matriz de confusão
  y_pred = np.argmax(y_pred, axis=1)
  cm = confusion_matrix(y_test, y_pred)




  # Plotar a matriz de confusão usando a biblioteca seaborn
  sns.heatmap(cm, annot=True, cmap='Blues')
  plt.get_current_fig_manager().window.state('zoomed')




  plt.show()

# Obter a acurácia do modelo nos dados de teste
accuracy = accuracy_score(y_test, y_pred)

# Imprimir a acurácia como uma porcentagem
print("Acurácia nos dados de teste: {:.2%}".format(accuracy))

duracao = fim_treinamento - inicio_treinamento # duração em segundos
print(f"O treinamento demorou {duracao:.6f} segundos para ser executado.")

duracao = (fim_teste - inicio_teste) / len(X_test) # duração em segundos
print(f"O teste de 1 padrão demorou {duracao:.6f} segundos para ser executado.")
