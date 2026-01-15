from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI(title="Wine Classification Prediction API")

# Load trained model
model = joblib.load("wine_quality_model.joblib")

# Define input schema (matches your CSV features exactly)
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    OD280_OD315_of_diluted_wines: float
    proline: float

    class Config:
        schema_extra = {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100.0,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "OD280_OD315_of_diluted_wines": 3.4,
                "proline": 1050.0
            }
        }

# Prediction endpoint
@app.post("/predict")
def predict_wine_class(features: WineFeatures):
    input_data = pd.DataFrame([features.dict()])

    input_data.rename(
        columns={"OD280_OD315_of_diluted_wines": "OD280/OD315_of_diluted_wines"},
        inplace=True
    )
    prediction = model.predict(input_data)

    return {"predicted_class": int(prediction[0])}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Classification Prediction API!"}
