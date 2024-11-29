from model import Model

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pymongo import MongoClient
import joblib
import numpy as np

app = FastAPI()
client = MongoClient('mongo', 27017)
db = client.test_database
collection = db.test_collection

model = joblib.load('model.pkl')

class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/add/{fruit}")
async def add_fruit(fruit: str):
    id = collection.insert_one({"fruit": fruit}).inserted_id 
    return {"id": str(id)}

@app.get("/list")
async def list_fruits():
    return {"results": list(collection.find({}, {"_id": False}))}

@app.post("/train")
def train_model():
    try:
        # Initialiser et entraîner le modèle
        model_instance = Model()
        best_params = model_instance.train()
        test_score = model_instance.evaluate()
        
        # Charger le modèle entraîné
        global model
        model = joblib.load('model.pkl')
        
        return {
            "message": "Model trained successfully!",
            "best_params": best_params,
            "test_score": test_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(item: Item):
    item_data = jsonable_encoder(item)
    features = np.array([[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}