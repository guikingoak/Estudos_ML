import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Lendo o Dataset
titanic_data = pd.read_excel('titanic.xlsx')

# 1 - Limpeza dos dados
titanic_data['Fare'] = titanic_data['Fare'].astype(str)
titanic_data['Fare'] = titanic_data['Fare'].str.replace(',', '.', regex=True)
titanic_data['Fare'] = titanic_data['Fare'].str.replace('[^0-9.]', '', regex=True)
titanic_data['Fare'] = titanic_data['Fare'].astype(float)
titanic_data['Age'] = pd.to_numeric(titanic_data['Age'], errors='coerce')
titanic_data.dropna(inplace=True)

# 2 - Transformando dados categóricos em numéricos
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Pclass'])

# Garantindo que todas as colunas esperadas existem
for col in ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male']:
    if col not in titanic_data.columns:
        titanic_data[col] = 0

# 3 - Separando os dados em conjuntos de treinamento e teste
X = titanic_data[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare', 'Sex_female', 'Sex_male']]
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4 - Escalonando os dados numéricos
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# 5 - Treinando o modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# 6 - Previsão dos dados
y_pred = model.predict(X_test)

# 7 - Resultados
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.4f}')
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

#modelo de regressão logística apresentou uma acurácia de 77% na previsão de sobreviventes do Titanic. Os resultados mostram que o modelo conseguiu um bom equilíbrio entre precisão e recall para ambas as classes