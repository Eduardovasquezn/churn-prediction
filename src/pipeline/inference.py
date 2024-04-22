import os
import sys

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(script_directory, '..', '..'))
sys.path.append(project_root)

from src.common.inference_utils import setup_model, CustomerData, add_missing_features
from src.common.data_preprocessing import preprocessing_functions
from src.common.logger import get_console_logger

import pandas as pd

import uvicorn
from fastapi import FastAPI

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

# app = FastAPI()

model, feature_names_from_model = setup_model()

# @app.post('/predict')
def predict_churn(item: CustomerData):
    logger.info("Get data")
    # data = item.dict()

    logger.info("Transform to df...")
    df = pd.DataFrame(item, index=[0])

    logger.info("Preprocessing...")
    df = preprocessing_functions(df=df)

    # missing columns levels train and test.
    add_missing_features(df=df, feature_names_from_model=feature_names_from_model)

    logger.info("Calculate predictions")
    prediction = model.predict(df[feature_names_from_model])[0]

    prediction_formatted = "Churn" if prediction == 1 else "Not Churn"
    logger.info(f"Prediction: {prediction_formatted}")

    return {"prediction": prediction_formatted}


if __name__ == '__main__':
    item = {
    'customerid': '3164-YAXFY',
    'gender': 'Male',
    'seniorcitizen': 0,
    'dependents': 'No',
    'partner': 'No',
    'contract': 'Month-to-month',
    'tenure': 57,
    'paymentmethod': 'Electronic check',
    'paperlessbilling': 'Yes',
    'monthlycharges': 53.75,
    'totalcharges': 3196.0,
    'datetime_x': '2021-01-23 18:03:34.711729620',
    'deviceprotection': 'Yes',
    'onlinebackup': 'No',
    'onlinesecurity': 'Yes',
    'internetservice': 'DSL',
    'multiplelines': 'No phone service',
    'phoneservice': 'No',
    'techsupport': 'No',
    'streamingmovies': 'Yes',
    'streamingtv': 'Yes',
    'datetime_y': '2021-01-23 18:03:34.711729620'
}
    prediction = predict_churn(item)

    # uvicorn.run(app, host='0.0.0.0', port=8080)
