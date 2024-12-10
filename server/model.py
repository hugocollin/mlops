from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class Model:
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, test_size=0.3, cv=5):
        
        # Chargement du jeu de données Iris
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        
        # Division des données en données d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=1
        )
        
        # Définition des hyperparamètres
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        
        # Initialisation du modèle RandomForestClassifier
        self.rf = RandomForestClassifier(random_state=1)
        
        # Initialisation de GridSearchCV avec les listes de paramètres
        self.grid_search = GridSearchCV(
            estimator=self.rf,
            param_grid={
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            },
            cv=self.cv,
            n_jobs=-1,
            verbose=2
        )
    
    def train(self):
        # Entraînement du modèle
        self.grid_search.fit(self.X_train, self.y_train)
        
        # Récupération des meilleurs paramètres
        best_params = self.grid_search.best_params_
        
        # Chemin absolu pour sauvegarder le modèle
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        
        # Sauvegarde du meilleur modèle
        joblib.dump(self.grid_search.best_estimator_, model_path)

        return best_params
    
    def evaluate(self):
        # Évaluation du modèle sur les données de test
        score = self.grid_search.score(self.X_test, self.y_test)
        
        return score
    
    def predict(self, X):
        # Chemin absolu pour charger le modèle sauvegardé
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        model = joblib.load(model_path)

        return model.predict(X)