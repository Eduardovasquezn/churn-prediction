import os
import sys

from pydantic import BaseModel

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(script_directory, '..', '..'))
sys.path.append(project_root)

from comet_ml import Experiment

from src.common.inference_features import add_missing_features, CustomerData
from src.common.data_preprocessing import preprocessing_functions
from src.common.paths import artifacts_path
from src.common.logger import get_console_logger
from src.common.model_registry import load_model_from_model_registry, get_feature_names_from_model

import uvicorn
import pandas as pd

from fastapi import FastAPI

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

comet_ml_api_key = os.getenv("COMET_ML_API_KEY", "")
comet_ml_workspace = os.getenv("COMET_ML_WORKSPACE", "")
comet_ml_model_name = os.getenv("COMET_ML_MODEL_NAME", "")
comet_ml_project_name = os.getenv("COMET_ML_PROJECT_NAME", "")

app = FastAPI()

logger.info("Specify Experiment")
experiment = Experiment(
    api_key=comet_ml_api_key,
    workspace=comet_ml_workspace,
    project_name=comet_ml_project_name
)

logger.info("Load model from Comet Model Registry")
model = load_model_from_model_registry(
    workspace=comet_ml_workspace,
    api_key=comet_ml_api_key,
    model_name=comet_ml_model_name,
)

# Feature names from model
artifact_path = artifacts_path()
feature_names_from_model = get_feature_names_from_model(experiment=experiment,
                                                        feature_names_directory_path=artifact_path)



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
