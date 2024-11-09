
import pandas as pd
import numpy as np
import kagglehub
import os

# EDA library
import matplotlib.pyplot as plt 
import seaborn as sns

# Pipeline 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

# MLflow library
import mlflow
#Save the model
import joblib

# Data capturing
path = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")
print("Path to dataset files:", path)

path = "/Users/cedric-omeryapo/.cache/kagglehub/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/versions/1"
energy_file = os.path.join(path, "energy_dataset.csv")
energy_dataset = pd.read_csv(energy_file)
energy_dataset['time'] = pd.to_datetime(energy_dataset['time'], utc=True)

# Preprocessing
energy_dataset_C = energy_dataset.drop(columns=[
    "generation hydro pumped storage aggregated",
    "forecast wind offshore eday ahead",
    "generation fossil coal-derived gas",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation marine",
    "generation wind offshore",
    "price day ahead"
])

# Renaming columns
energy_dataset_C.columns = energy_dataset_C.columns.str.replace(' ', '_').str.replace('-', '_')
energy_dataset_C.columns = energy_dataset_C.columns.str.replace(' ','_').str.replace('/','_')
energy_dataset_C.set_index("time", inplace=True)

# Sort index and add season column
energy_dataset_C = energy_dataset_C.sort_index()

# Set conditions for seasons
condition_winter = (energy_dataset_C.index.month >= 1) & (energy_dataset_C.index.month <= 3)
condition_spring = (energy_dataset_C.index.month >= 4) & (energy_dataset_C.index.month <= 6)
condition_summer = (energy_dataset_C.index.month >= 7) & (energy_dataset_C.index.month <= 9)
condition_autumn = (energy_dataset_C.index.month >= 10) & (energy_dataset_C.index.month <= 12)

# Create season column
energy_dataset_C['season'] = np.where(condition_winter, 'winter',
                        np.where(condition_spring, 'spring',
                                     np.where(condition_summer, 'summer',
                                              np.where(condition_autumn, 'autumn', np.nan))))

# MLflow setup
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("ML project energie")

# Start MLflow run
with mlflow.start_run():
    y = energy_dataset_C['price_actual']
    X = energy_dataset_C.drop(columns='price_actual')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    # Model pipeline
    model = HistGradientBoostingRegressor(learning_rate= 0.09070280434358921, max_iter=547, max_leaf_nodes= 181)

    # Preprocessing pipeline
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Define numerical columns for preprocessing
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                        X_train[cname].dtype == "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', onehot_encoder, categorical_cols),  
            ('num', SimpleImputer(strategy='median'), numerical_cols)  
        ],
        remainder='passthrough'  
    )

    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    my_pipeline.fit(X_train, y_train)
    
    # Predicting with the model
    y_pred = my_pipeline.predict(X_test)

    # Evaluate metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    scores = my_pipeline.score(X_train, y_train)
    val_scores = cross_val_score(my_pipeline, X_train, y_train, cv=5)
    mean_val_score = val_scores.mean()
    std_val_score = val_scores.std()
    
    # Log metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", scores)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("mean_val_score", mean_val_score)
    mlflow.log_metric("std_val_score", std_val_score)

    # Calculate and print baseline values
    mean_pred = y_pred.mean()
    print('Mean Price Per KW/h Pred:', mean_pred)
    print('-------------------------------------------------------------------')
    print('RMSE:', rmse)

# Function to predict prices based on user input
def predict_price(features):
    """ Predict price given the input features as a dictionary. """
    input_data = pd.DataFrame([features])
    prediction = my_pipeline.predict(input_data)
    return prediction[0]
print(X_train.columns)

# Example usage of the predict function
if __name__ == "__main__":
    # Example input features
    features = {
        'generation_biomass': 150,
        'generation_fossil_brown_coal_lignite': 200,
        'generation_fossil_gas': 100,
        'generation_fossil_hard_coal': 150,
        'generation_fossil_oil': 30,
        'generation_hydro_pumped_storage_consumption': 40,
        'generation_hydro_run_of_river_and_poundage': 50,
        'generation_hydro_water_reservoir': 0,
        'generation_nuclear': 0,
        'generation_other': 0,
        'generation_other_renewable': 70,
        'generation_solar': 10,
        'generation_waste': 30,
        'generation_wind_onshore': 50,
        'forecast_solar_day_ahead': 10,
        'forecast_wind_onshore_day_ahead': 0,
        'total_load_forecast': 1500,
        'total_load_actual': 1500,
        'season': 'summer'  # Adjust based on the conditions
    }

    predicted_price = predict_price(features)
    print(f"Predicted price: {predicted_price:.2f}")



# After training the model
joblib.dump(my_pipeline, 'best_model.pkl')
