
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

# Carregar e pré-processar os dados
df = pd.read_excel('titanic.xlsx')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df_cleaned = df.dropna(subset=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])

label_encoder = LabelEncoder()
df_cleaned['Sex'] = label_encoder.fit_transform(df_cleaned['Sex'])

features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
X = df_cleaned[features]
y = df_cleaned['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Função para treinar e avaliar modelos
def evaluate_model(model, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Modelo: {model_name}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Relatório de Classificação:\n{report}")
    print(f"Tempo de Treinamento: {train_time:.4f}s")
    print(f"Tempo de Previsão: {predict_time:.4f}s")
    print("-" * 50)

# Avaliar KNN
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, "KNN")

# Avaliar Regressão Logística
logistic_model = LogisticRegression(random_state=42)
evaluate_model(logistic_model, "Regressão Logística")

# Avaliar Árvore de Decisão
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=3)
evaluate_model(decision_tree, "Árvore de Decisão")



"""

Com base nos resultados obtidos:

- **KNN**:
 Acurácia: 0.7541
  - F1-Score: 0.75 (média ponderada)
  - O KNN apresenta desempenho inferior em relação aos outros modelos, com menor acurácia e F1-score. Ele é mais sensível à dimensionalidade e ruídos nos dados.

- **Regressão Logística**:
     - Acurácia: 0.8033
  - F1-Score: 0.80 (média ponderada)
  - A Regressão Logística apresenta bom desempenho, com métricas equilibradas e o menor tempo de previsão, sendo uma boa escolha para problemas de classificação binária.

- **Árvore de Decisão**:
  - Acurácia: 0.8197
  - F1-Score: 0.82 (média ponderada)
  - A Árvore de Decisão apresenta a melhor acurácia e F1-score, além de ser interpretável, o que a torna uma excelente escolha, especialmente se a interpretabilidade for uma prioridade.

**Conclusão**:
- A Árvore de Decisão é o modelo com melhor desempenho geral, com a maior acurácia e F1-score.
- A Regressão Logística é uma boa alternativa, com desempenho próximo.
- O KNN é menos adequado devido ao seu desempenho inferior e maior sensibilidade a ruídos e dimensionalidade.
"""