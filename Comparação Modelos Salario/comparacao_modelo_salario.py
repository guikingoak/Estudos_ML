import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Carregamento e limpeza dos dados
df = pd.read_csv('salarios_recife_outubro_2013.csv')
df['R$SALARIO'] = df['R$SALARIO'].replace({'R\$': '', ',': ''}, regex=True).astype(float)

# 2. Tratamento de valores ausentes
df['FUNCAO'].fillna(df['FUNCAO'].mode()[0], inplace=True)

# 3. Codificação de variáveis categóricas
categorical_cols = ['ORGAO', 'NOME', 'CATEGORIA', 'CARGO', 'FUNCAO']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Separação das features e target
X = df.drop(columns=['NOME', 'R$SALARIO'])
y = df['R$SALARIO']

# 5. Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 6. Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Treinamento dos modelos
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)

lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# 8. Predições
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# 9. Avaliação dos modelos
def print_metrics(y_true, y_pred, model_name):
    print(f"{model_name} - Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"{model_name} - Mean Squared Error: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"{model_name} - R-squared: {r2_score(y_true, y_pred):.4f}\n")

print_metrics(y_test, y_pred_rf, "Random Forest")
print_metrics(y_test, y_pred_lr, "Linear Regression")

#Random Forest - Mean Absolute Error: 499.71
#Random Forest - Mean Squared Error: 836968.69
#Random Forest - R-squared: 0.9145

#Linear Regression - Mean Absolute Error: 1613.47
#Linear Regression - Mean Squared Error: 9457852.09
#Linear Regression - R-squared: 0.0337


#Random Forest teve desempenho muito superior: 
# Erros (MAE e MSE) muito menores.
# R² próximo de 1, indicando que o modelo explica mais de 91% da variância dos salários.


#  Linear Regression teve desempenho ruim:
# Erros altos.
#R² próximo de zero, mostrando que quase não explica a variação dos salários.


#Conclusão
#O Random Forest é claramente o melhor modelo para este conjunto de dados, capturando padrões que a regressão linear não consegue.
#Isso sugere que as relações entre as variáveis e o salário são complexas e não lineares, o que é bem modelado pelo Random Forest.

