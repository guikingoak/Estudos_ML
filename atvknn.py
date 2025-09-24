import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# =========================
# Exploração dos Dados
# =========================
df = pd.read_csv("ytktk.csv")


# =========================
# Pré-processamento
# =========================
features = [ 'views','completion_rate',]
target = 'region'

# Preenchendo valores nulos das features com a média
df[features] = df[features].fillna(df[features].mean())

# Transformando variáveis categóricas em dummies
X = pd.get_dummies(df[features], drop_first=True)
y = df[target]

# =========================
# Divisão dos Dados
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# =========================
# Treinamento de Modelo
# =========================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# =========================
# Avaliação do Modelo
# =========================
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Comparação com modelo Dummy
dummy = DummyClassifier(strategy='uniform')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(f"Dummy Accuracy: {accuracy_score(y_test, dummy_pred):.2f}")

# =========================
# Visualização da Fronteira de Decisão
# =========================
h = 0.02 

X_vis = X[['views', 'completion_rate']]
x_min, x_max = X_vis.iloc[:, 0].min() - h, X_vis.iloc[:, 0].max() + h
y_min, y_max = X_vis.iloc[:, 1].min() - h, X_vis.iloc[:, 1].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_vis.iloc[:, 0], y=X_vis.iloc[:, 1], hue=y, style=y, palette='deep', s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=3)")

buffer = StringIO()
plt.savefig(buffer, format='png')
print(buffer.getvalue())

# =========================
# Relatório Final
# =========================
'''Conclusão: Foram testadas várias combinações de features e a combinação de 'views' e 'completion_rate' apresentou a melhor acurácia, superando o modelo Dummy por uma diferença de apenas 0.01. 
Esse resultado indica que, para este dataset, o modelo KNN é bastante afetado por ruídos e outliers presentes nos dados, o que prejudica sua capacidade de separação entre as classes. 
Recomenda-se experimentar outros algoritmos menos sensíveis a ruídos para tentar melhorar a performance.
'''