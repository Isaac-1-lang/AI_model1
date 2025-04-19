# Rwanda Crop Yield Prediction Model

This project implements a machine learning model to predict crop yields in Rwanda based on historical data and environmental factors.

## Features

- Predicts yields for multiple crops (Maize, Rice, Beans, Cassava)
- Covers all major regions of Rwanda (Kigali, Eastern, Northern, Southern, Western)
- Takes into account environmental factors (Rainfall, Average Temperature)
- Uses historical data from 2005-2024

## Project Structure

```
.
├── data/
│   └── rwanda_crop_training_data_2005_2024.csv
├── models/
│   ├── crop_yield_model.joblib
│   ├── crop_encoder.joblib
│   ├── region_encoder.joblib
│   └── feature_scaler.joblib
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

## Model Details

The model uses a Random Forest Regressor with the following parameters:
- Number of trees: 100
- Maximum depth: 10
- Minimum samples for split: 5
- Minimum samples per leaf: 2

## Data

The training data includes:
- Time span: 2005-2024
- Crops: Maize, Rice, Beans, Cassava
- Regions: Kigali, Eastern, Northern, Southern, Western
- Features: Rainfall, Average Temperature, Yield

## License

This project is licensed under the MIT License.
