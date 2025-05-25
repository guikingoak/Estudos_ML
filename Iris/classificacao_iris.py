
import sklearn
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    if name == "KNN":
        best_k = 1
        best_score = 0
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
        model = KNeighborsClassifier(n_neighbors=best_k)

    model.fit(X_train_scaled if name == "KNN" else X_train, y_train)
    y_pred = model.predict(X_test_scaled if name == "KNN" else X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, target_names=iris.target_names),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

results_summary = pd.DataFrame({model: {"Accuracy": metrics["Accuracy"]} for model, metrics in results.items()})
classification_reports = {model: metrics["Classification Report"] for model, metrics in results.items()}
confusion_matrices = {model: metrics["Confusion Matrix"] for model, metrics in results.items()}

pprint(results_summary)
pprint(classification_reports)
pprint(confusion_matrices)

"""Análise Comparativa:
Árvore de Decisão:
Teve a melhor performance geral, com uma acurácia de 95%. Confunde poucas amostras e mantém um bom equilíbrio entre precisão e recall.

KNN:
Teve a performance de 93,33% de acurácia. Apresentou uma maior dificuldade na classificação em relação a árvore de decisão com uma das classes.

Random Forest:
Teve a performance de 91,67% de acurácia.Não foi tão eficiente no conjunto de teste em relação aos outros dois modelos.


Conclusão:
A Árvore de Decisão parece ser a melhor escolha neste caso. Ela teve a maior acurácia e conseguiu manter um desempenho equilibrado para todas as classes. Além disso, é o modelo com maior praticidade em interpretar seus dados e costuma funcionar com maior facilidades ao utilizar conjuntos de dados menores como o Iris.
"""