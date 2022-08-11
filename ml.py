from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


# Altere o caminho do arqwuivo csv, ou busque outro modelo de alguma API
path = 'wine_dataset.csv'
arquivo = pd.read_csv(path)

#Refatora os tipos de vinho
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

# X recebe apenas o tipo enquanto Y recebe todas outras colunas
x = arquivo['style']
y = arquivo.drop('style', axis=1)

# Set as variaveis de teste e treino junto a funcao call
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# # Variavel recebe a funcao algoritmo de arvore binaria e inicia o treino das colunas
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)
#
#Mostra resultado teste das colunas
resultado = modelo.score(x_teste, y_teste)
print("Acurácia: ", resultado)

# Cria a previsao das linha 300 a 600
previsoes = modelo.predict(x_teste[300:600])

print(previsoes)
