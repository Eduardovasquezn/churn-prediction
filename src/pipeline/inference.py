import os
import sys

from pydantic import BaseModel

from src.common.inference_utils import setup_model, CustomerData, add_missing_features

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(script_directory, '..', '..'))
sys.path.append(project_root)


from src.common.data_preprocessing import preprocessing_functions

from src.common.logger import get_console_logger

import uvicorn
import pandas as pd

from fastapi import FastAPI

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

app = FastAPI()

model, feature_names_from_model = setup_model()

@app.post('/predict')
def predict_churn(item: CustomerData):
    logger.info("Get data")
    data = item.dict()

    logger.info("Transform to df...")
    df = pd.DataFrame(data, index=[0])

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
    uvicorn.run(app, host='0.0.0.0', port=8080)
