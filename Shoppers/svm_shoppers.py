

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df_original = pd.read_csv('online_shoppers_intention.csv')
df_original


"""Converter Colunas Categoricas:"""

df_original.dropna(inplace=True)

categorical_cols = ['Month', 'VisitorType', 'Weekend', 'Revenue']
for col in categorical_cols:
  df_original[col] = LabelEncoder().fit_transform(df_original[col])

"""Dividir os dados:"""

X = df_original.drop('Revenue', axis=1)
y = df_original['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Ajuste test_size e random_state se desejar

"""Escalonar os dados:"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""Treinando modelo SVM"""

svm_model = SVC(kernel='linear')  # Experimente outros kernels: 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)

"""Avaliando o modelo"""

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Matriz de confusão:\n{conf_matrix}")

"""Ajuste de Hiperparametros"""

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}  # Defina os parâmetros para otimizar
grid_search = GridSearchCV(svm_model, param_grid, cv=5)  # cv é o número de folds na validação cruzada
grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")

#Apesar do bom desempenho, o modelo apresenta mais erros ao identificar visitantes que realmente geram receita. Isso pode indicar um leve desbalanceamento ou dificuldade do modelo em identificar esse grupo.