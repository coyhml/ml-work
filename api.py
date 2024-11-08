from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib  # To load the model if saved as a joblib file



# Load the model pipeline
model_pipeline = joblib.load('best_model.pkl')  # Make sure to save your model pipeline previously

# Create a FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    generation_biomass: float
    generation_fossil_brown_coal_lignite: float
    generation_fossil_gas: float
    generation_fossil_hard_coal: float
    generation_fossil_oil: float
    generation_hydro_pumped_storage_consumption: float
    generation_hydro_run_of_river_and_poundage: float
    generation_hydro_water_reservoir: float
    generation_nuclear: float
    generation_other: float
    generation_other_renewable: float
    generation_solar: float
    generation_waste: float
    generation_wind_onshore: float
    forecast_solar_day_ahead: float
    forecast_wind_onshore_day_ahead: float
    total_load_forecast: float
    total_load_actual: float
    season: str

# Define a route for prediction
@app.post("/predict")
def predict(input: PredictionInput):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([input.dict()])
    
    # Make predictions using the model
    prediction = model_pipeline.predict(input_data)
    
    return {"predicted_price": prediction[0]}

# To run the application, use the following command in the terminal:
# uvicorn api:app --reload
