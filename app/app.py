from fastapi import FastAPI, Form
import pickle
import pandas as pd
import os

app = FastAPI()


# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, "models", "model.pkl"), "rb"))
target_encoder = pickle.load(open(os.path.join(BASE_DIR, "models", "target_encoder.pkl"), "rb"))


# HOME ROUTE
@app.get("/")
def home():
    return {"message": "Tourism Accommodation Grade Prediction API Running"}


# PREDICTION ROUTE
@app.post("/predict")
def predict(Rooms: float = Form(...),
            Latitude: float = Form(...),
            Longitude: float = Form(...),
            District: str = Form(...),
            Type: str = Form(...)):

    data = pd.DataFrame([[Rooms, Latitude, Longitude, District, Type]],
                        columns=['Rooms', 'Latitude', 'Longitude', 'District', 'Type'])

    prediction = model.predict(data)[0]
    decoded_prediction = target_encoder.inverse_transform([prediction])[0]

    return {"Predicted Level": decoded_prediction}
