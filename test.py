# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


import kagglehub

path = kagglehub.dataset_download("prajwaldongre/top-dog-breeds-around-the-world")
print("Path to dataset files:", path)

data_path = f"{path}/Dog Breads Around The World.csv"  # Remplacez par le nom du fichier
df = pd.read_csv(data_path)

print(df.head())


df = pd.get_dummies(df)

# Séparer les variables
X = df.drop("Training Difficulty (1-10)", axis=1)  # Remplacez par le nom correct de la colonne cible
y = df["Training Difficulty (1-10)"]

# Set DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))