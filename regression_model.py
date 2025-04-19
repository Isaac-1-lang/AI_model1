import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import math
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
train_file = "data/rwanda_crop_training_data_2005_2024.csv"
model_file = "models/crop_yield_model.pkl"
scaler_file = "models/scaler.pkl"
encoder_file = "models/encoder.pkl"

# Verify file existence
if not os.path.exists(train_file):
    logging.error(f"File not found: {train_file}")
    raise FileNotFoundError(f"Ensure {train_file} is in the data/ folder")

# Load data
try:
    train_data = pd.read_csv(train_file)
    logging.info("Data loaded successfully")
    logging.info(f"Training data columns: {train_data.columns.tolist()}")
    logging.info(f"Training data shape: {train_data.shape}")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Fix negative rainfall in raw data
train_data['Rainfall(mm)'] = train_data['Rainfall(mm)'].clip(lower=0)

# Preprocess data
numerical_cols = ['Rainfall(mm)', 'Avg_Temperature(°C)']
imputer = SimpleImputer(strategy='mean')
train_data[numerical_cols] = imputer.fit_transform(train_data[numerical_cols])

# Cap outliers with non-negative lower bound
def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(Q1 - 1.5 * IQR, 0)
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

train_data = cap_outliers(train_data, numerical_cols)

# Scale features
scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
crop_encoded = encoder.fit_transform(train_data[['Crop', 'Region']])
crop_df = pd.DataFrame(crop_encoded, columns=encoder.get_feature_names_out(['Crop', 'Region']))

# Combine features
X = pd.concat([train_data[numerical_cols], crop_df], axis=1)
y = train_data['Yield(t/ha)']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
logging.info(f"Best parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = model.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
logging.info("\nModel Evaluation Results:")
logging.info(f"RMSE: {rmse:.4f}")
logging.info(f"R2 Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
logging.info(f"Cross-validated RMSE: {cv_rmse:.4f}")

# Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/images/feature_importance.png')

# Save model and preprocessing objects
joblib.dump(model, model_file)
joblib.dump(scaler, scaler_file)
joblib.dump(encoder, encoder_file)
logging.info("Model and preprocessing objects saved")