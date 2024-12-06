from model import Model

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, conint
from pymongo import MongoClient
import joblib
import numpy as np
from typing import List

# Initialisation de l'application FastAPI
app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection

# Chargement du modèle sauvegardé
try:
    model = joblib.load('model.pkl')
except:
    model = None

# Définition des contraintes pour les paramètres d'entraînement
ConInt10_1000 = conint(ge=10, le=1000)
ConInt1_100 = conint(ge=1, le=100)
ConInt2_50 = conint(ge=2, le=50)
ConInt1_50 = conint(ge=1, le=50)

# Définition de la classe des paramètres d'entraînement du modèle
class TrainParams(BaseModel):
    n_estimators: List[ConInt10_1000] = Field(..., description="Nombre d'estimateurs pour le Random Forest")
    max_depth: List[ConInt1_100] = Field(..., description="Profondeur maximale des arbres")
    min_samples_split: List[ConInt2_50] = Field(..., description="Nombre minimum d'échantillons requis pour diviser un nœud")
    min_samples_leaf: List[ConInt1_50] = Field(..., description="Nombre minimum d'échantillons requis à une feuille")
    test_size: float = Field(0.3, gt=0.0, lt=1.0, description="Fraction des données pour le test")
    cv: int = Field(5, ge=2, le=10, description="Nombre de folds pour la validation croisée")

# Définition de la classe des paramètres des caractéristiques d'une fleur
class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Définition des routes de l'API
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Ajout d'un fruit dans la base de données
@app.get("/add/{fruit}")
async def add_fruit(fruit: str):
    id = collection.insert_one({"fruit": fruit}).inserted_id 
    return {"id": str(id)}

# Récupération de la liste des fruits
@app.get("/list")
async def list_fruits():
    return {"results": list(collection.find({}, {"_id": False}))}

# Entraînement du modèle
@app.post("/train")
def train_model(params: TrainParams):
    try:
        # Initialisation du modèle avec les paramètres d'entraînement
        model_instance = Model(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            test_size=params.test_size,
            cv=params.cv
        )
        best_params = model_instance.train()
        test_score = model_instance.evaluate()
        
        # Chargement du meilleur modèle
        global model
        model = joblib.load('model.pkl')
        
        return {
            "message": "Model trained successfully !",
            "best_params": best_params,
            "test_score": test_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prédiction de l'espèce d'une fleur
@app.post("/predict")
def predict(item: Item):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    item_data = jsonable_encoder(item)
    features = np.array([[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}