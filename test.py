import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import kagglehub
import matplotlib.pyplot as plt

# Chargement et préparation des données
path = kagglehub.dataset_download("prajwaldongre/top-dog-breeds-around-the-world")
data_path = f"{path}/Dog Breads Around The World.csv"
df = pd.read_csv(data_path)
df = pd.get_dummies(df)

X = df.drop("Training Difficulty (1-10)", axis=1)
y = df["Training Difficulty (1-10)"]

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(i) for i in y.unique()], filled=True, rounded=True)
plt.title("Arbre de Décision pour la Difficulté d'Entraînement des Races de Chiens")
plt.savefig("arbre_decision.png")
plt.show()

# Optimisation des hyperparamètres
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score de validation croisée :", grid_search.best_score_)

# Importance des caractéristiques
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importance des caractéristiques :\n", feature_importance)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Importance des Caractéristiques")
plt.xlabel("Caractéristiques")
plt.ylabel("Importance")
plt.show()

# Courbe Précision-Rappel
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Précision-Rappel")
plt.show()

# Représentation textuelle de l'arbre de décision
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
