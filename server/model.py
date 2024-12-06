from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

class Model:
    def __init__(self, param_grid=None, test_size=0.3, cv=5):
        
        # Chargement du jeu de données Iris
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        
        # Division des données en données d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=1
        )
        
        # Définition de la grille des hyperparamètres
        if param_grid is None:
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            self.param_grid = param_grid
        
        # Initialisation du modèle RandomForestClassifier
        self.rf = RandomForestClassifier(random_state=1)
        
        # Initialisation de GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=self.rf,
            param_grid=self.param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
    
    def train(self):
        # Entraînement du modèle
        self.grid_search.fit(self.X_train, self.y_train)
        
        # Récupération des meilleurs paramètres
        best_params = self.grid_search.best_params_
        
        # Sauvegarde du meilleur modèle
        joblib.dump(self.grid_search.best_estimator_, 'model.pkl')

        return best_params
    
    def evaluate(self):
        # Évaluation du modèle sur les données de test
        score = self.grid_search.score(self.X_test, self.y_test)
        
        return score
    
    def predict(self, X):
        # Chargement du modèle sauvegardé
        model = joblib.load('model.pkl')
        return model.predict(X)