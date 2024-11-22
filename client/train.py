from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger le jeu de données Iris
iris = load_iris()
X, y = iris.data, iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné dans un fichier model.pkl
joblib.dump(model, '../server/model.pkl')
print("Modèle entraîné et sauvegardé dans model.pkl")