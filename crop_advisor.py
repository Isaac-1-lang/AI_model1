from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load the trained model and preprocessing objects
try:
    model = joblib.load('models/crop_yield_model.joblib')
    scaler = joblib.load('models/feature_scaler.joblib')
    crop_encoder = joblib.load('models/crop_encoder.joblib')
    region_encoder = joblib.load('models/region_encoder.joblib')
    print("Model and preprocessing objects loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_dummy = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    crop_encoder = LabelEncoder()
    crop_encoder.fit(['Maize', 'Rice', 'Beans', 'Cassava'])
    region_encoder = LabelEncoder()
    region_encoder.fit(['Kigali', 'Eastern', 'Northern', 'Southern', 'Western'])

# Load training data for visualization
try:
    train_data = pd.read_csv('data/rwanda_crop_training_data_2005_2024.csv')
    print("Training data loaded successfully")
except Exception as e:
    print(f"Error loading training data: {e}")
    train_data = pd.DataFrame({
        'Year': [2020, 2021, 2022],
        'Crop': ['Maize', 'Rice', 'Beans'],
        'Region': ['Kigali', 'Eastern', 'Northern'],
        'Yield(t/ha)': [2.5, 3.0, 2.0]
    })

# Create a file to store predictions
PREDICTIONS_FILE = 'data/saved_predictions.json'
if not os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump([], f)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chart-data')
def get_chart_data():
    try:
        # Dummy data as default
        dummy_data = {
            'yieldTrend': {
                'labels': [2020, 2021, 2022, 2023, 2024],
                'datasets': [
                    {
                        'label': 'Maize',
                        'data': [2.5, 2.7, 2.4, 2.8, 3.0],
                        'borderColor': 'rgb(255, 215, 0)',
                        'fill': False
                    },
                    {
                        'label': 'Rice',
                        'data': [3.0, 3.2, 3.1, 3.3, 3.5],
                        'borderColor': 'rgb(0, 128, 0)',
                        'fill': False
                    }
                ]
            },
            'regionalDistribution': {
                'labels': ['Kigali', 'Eastern', 'Northern', 'Southern', 'Western'],
                'data': [2.5, 3.0, 2.8, 2.6, 2.7]
            },
            'featureImportance': {
                'labels': ['Year', 'Rainfall', 'Temperature', 'Crop', 'Region'],
                'data': [0.2, 0.3, 0.25, 0.15, 0.1]
            },
            'recentPredictions': [
                {
                    'timestamp': '2025-04-19 10:00:00',
                    'crop': 'Maize',
                    'region': 'Kigali',
                    'rainfall': 800,
                    'temperature': 23.5,
                    'prediction': 2.75
                }
            ]
        }
        
        # Try to include real data if available
        if not train_data.empty:
            yield_trend_data = {
                'labels': train_data['Year'].unique().tolist(),
                'datasets': []
            }
            for crop in train_data['Crop'].unique():
                crop_data = train_data[train_data['Crop'] == crop]
                yield_trend_data['datasets'].append({
                    'label': crop,
                    'data': crop_data.groupby('Year')['Yield(t/ha)'].mean().tolist(),
                    'borderColor': get_color_for_crop(crop),
                    'fill': False
                })
            regional_data = train_data.groupby('Region')['Yield(t/ha)'].mean()
            regional_distribution = {
                'labels': regional_data.index.tolist(),
                'data': regional_data.values.tolist()
            }
            feature_names = ['Year', 'Rainfall', 'Temperature', 'Crop', 'Region']
            feature_importance = {
                'labels': feature_names,
                'data': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else [0] * len(feature_names)
            }
            with open(PREDICTIONS_FILE, 'r') as f:
                predictions = json.load(f)
            recent_predictions = predictions[-10:]
            return jsonify({
                'yieldTrend': yield_trend_data,
                'regionalDistribution': regional_distribution,
                'featureImportance': feature_importance,
                'recentPredictions': recent_predictions
            })
        else:
            print("Using dummy chart data due to empty training data")
            return jsonify(dummy_data)
    except Exception as e:
        print(f"Error preparing chart data: {e}")
        return jsonify(dummy_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        current_year = datetime.now().year
        input_data = pd.DataFrame({
            'Year': [current_year],
            'Crop': [data['crop']],
            'Region': [data['region']],
            'Rainfall(mm)': [float(data['rainfall'])],
            'Avg_Temperature(°C)': [float(data['temperature'])]
        })
        if 'soilType' in data and data['soilType']:
            print(f"Soil type provided: {data['soilType']}")
        crop_encoded = crop_encoder.transform(input_data['Crop'])
        region_encoded = region_encoder.transform(input_data['Region'])
        input_data['Crop'] = crop_encoded
        input_data['Region'] = region_encoded
        input_data = input_data.rename(columns={
            'Crop': 'Crop_encoded',
            'Region': 'Region_encoded'
        })
        features = ['Year', 'Crop_encoded', 'Region_encoded', 'Rainfall(mm)', 'Avg_Temperature(°C)']
        X = input_data[features]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prediction = prediction * (1 + np.random.normal(0, 0.05))
        save_prediction(data, prediction)
        session['prediction_result'] = {
            'prediction': float(prediction),
            'message': f'Expected harvest: {prediction:.2f} tons/hectare',
            'crop': data['crop'],
            'region': data['region'],
            'rainfall': float(data['rainfall']),
            'temperature': float(data['temperature']),
            'soilType': data.get('soilType', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify({
            'success': True,
            'redirect': url_for('results')
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/save_prediction', methods=['POST'])
def save_prediction_route():
    try:
        data = request.get_json()
        save_prediction(data, data['prediction'])
        return jsonify({'success': True, 'message': 'Prediction saved successfully'})
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/results')
def results():
    prediction_result = session.get('prediction_result', None)
    if not prediction_result:
        return redirect(url_for('home'))
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
        recent_predictions = predictions[-10:]  # Last 10 predictions
    except Exception as e:
        print(f"Error loading predictions: {e}")
        recent_predictions = [
            {
                'timestamp': '2025-04-19 10:00:00',
                'crop': 'Maize',
                'region': 'Kigali',
                'rainfall': 800,
                'temperature': 23.5,
                'prediction': 2.75
            }
        ]
    return render_template('results.html', result=prediction_result, recent_predictions=recent_predictions)

def save_prediction(data, prediction):
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
        predictions.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'crop': data['crop'],
            'region': data['region'],
            'rainfall': float(data['rainfall']),
            'temperature': float(data['temperature']),
            'soilType': data.get('soilType', ''),
            'prediction': float(prediction)
        })
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Prediction saved: {prediction:.2f} tons/hectare for {data['crop']} in {data['region']}")
    except Exception as e:
        print(f"Error saving prediction: {e}")

def get_color_for_crop(crop):
    colors = {
        'Maize': 'rgb(255, 215, 0)',
        'Rice': 'rgb(0, 128, 0)',
        'Beans': 'rgb(139, 69, 19)',
        'Cassava': 'rgb(218, 165, 32)'
    }
    return colors.get(crop, 'rgb(128, 128, 128)')

if __name__ == '__main__':
    app.run(debug=True)