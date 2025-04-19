import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('data/rwanda_crop_training_data_2005_2024.csv')
    
    # Convert numerical columns to numeric, coercing errors to NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Rainfall(mm)'] = pd.to_numeric(df['Rainfall(mm)'], errors='coerce')
    df['Avg_Temperature(°C)'] = pd.to_numeric(df['Avg_Temperature(°C)'], errors='coerce')
    df['Yield(t/ha)'] = pd.to_numeric(df['Yield(t/ha)'], errors='coerce')
    
    # Handle NaN values by filling with median
    if df['Year'].isna().any():
        print("Warning: Some 'Year' values could not be converted to numeric. Filling with median year.")
        df['Year'] = df['Year'].fillna(df['Year'].median())
    if df['Rainfall(mm)'].isna().any():
        print("Warning: Some 'Rainfall(mm)' values could not be converted to numeric. Filling with median.")
        df['Rainfall(mm)'] = df['Rainfall(mm)'].fillna(df['Rainfall(mm)'].median())
    if df['Avg_Temperature(°C)'].isna().any():
        print("Warning: Some 'Avg_Temperature(°C)' values could not be converted to numeric. Filling with median.")
        df['Avg_Temperature(°C)'] = df['Avg_Temperature(°C)'].fillna(df['Avg_Temperature(°C)'].median())
    if df['Yield(t/ha)'].isna().any():
        print("Warning: Some 'Yield(t/ha)' values could not be converted to numeric. Filling with median.")
        df['Yield(t/ha)'] = df['Yield(t/ha)'].fillna(df['Yield(t/ha)'].median())
    
    # Debug: Inspect the data
    print("\n=== Data Inspection ===")
    print("Yield(t/ha) Distribution:")
    print(df['Yield(t/ha)'].describe())
    print("\nRainfall(mm) Distribution:")
    print(df['Rainfall(mm)'].describe())
    print("\nAvg_Temperature(°C) Distribution:")
    print(df['Avg_Temperature(°C)'].describe())
    print("\nYield(t/ha) for High Rainfall (>1000 mm):")
    print(df[df['Rainfall(mm)'] > 1000]['Yield(t/ha)'].describe())
    print("\nYield(t/ha) for Maize in Kigali:")
    print(df[(df['Crop'] == 'Maize') & (df['Region'] == 'Kigali')]['Yield(t/ha)'].describe())
    
    # Create label encoders for categorical variables
    le_crop = LabelEncoder()
    le_region = LabelEncoder()
    
    # Transform categorical variables
    df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])
    df['Region_encoded'] = le_region.fit_transform(df['Region'])
    
    # Save label encoders for later use
    joblib.dump(le_crop, 'models/crop_encoder.joblib')
    joblib.dump(le_region, 'models/region_encoder.joblib')
    
    # Select features for training (5 features total)
    features = ['Year', 'Crop_encoded', 'Region_encoded', 'Rainfall(mm)', 'Avg_Temperature(°C)']
    target = 'Yield(t/ha)'
    
    X = df[features]
    y = df[target]
    
    # Verify data types of features
    print("\nData types of features:\n", X.dtypes)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"RMSE: {rmse:.2f} t/ha")
    print(f"R2 Score: {r2:.2f}")
    
    return rmse, r2

def save_model(model, rmse, r2):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/crop_yield_model.joblib')
    
    # Save model metrics
    metrics = {
        'rmse': rmse,
        'r2_score': r2
    }
    joblib.dump(metrics, 'models/model_metrics.joblib')
    
    print("\nModel and metrics saved successfully!")

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    rmse, r2 = evaluate_model(model, X_test, y_test)
    
    print("\nSaving model and metrics...")
    save_model(model, rmse, r2)

if __name__ == "__main__":
    main()