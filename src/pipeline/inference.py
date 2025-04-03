from src.common import setup_model, CustomerData, add_missing_features
from src.common import preprocessing_functions
from src.common import get_console_logger

import pandas as pd

import uvicorn
from fastapi import FastAPI

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

app = FastAPI()

model, feature_names_from_model = setup_model()


@app.post("/predict")
def predict_churn(data: CustomerData):
    logger.info("Get data")
    data = data.model_dump()

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
