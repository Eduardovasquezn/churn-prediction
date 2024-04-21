import pandas as pd
import os
import sys

from pydantic import BaseModel

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(script_directory, '..', '..'))
sys.path.append(project_root)

from comet_ml import Experiment, ExistingExperiment

from src.common.data_preprocessing import preprocessing_functions
from src.common.paths import artifacts_path

import uvicorn
import pandas as pd

from fastapi import FastAPI
from src.common.logger import get_console_logger
from src.common.model_registry import load_model_from_model_registry, get_feature_names_from_model, \
    get_feature_names_from_model1, get_feature_names_from_model2
from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

logger.info("Specify Comet ML experiment")
comet_ml_api_key = os.getenv("COMET_ML_API_KEY", "")
comet_ml_workspace = os.getenv("COMET_ML_WORKSPACE", "")
comet_ml_model_name = os.getenv("COMET_ML_MODEL_NAME", "")
comet_ml_project_name = os.getenv("COMET_ML_PROJECT_NAME", "")



# test = get_feature_names_from_model1(api_key=comet_ml_api_key, version="1.0.0")
exp = ExistingExperiment(api_key=comet_ml_api_key,workspace=comet_ml_workspace)
test = get_feature_names_from_model1(api_key=comet_ml_api_key, version="1.0.0")









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

params = model.get_params()
logger.info(params)

logger.info("Load transformation functions")
artifact_path = artifacts_path()

from src.common.data_preprocessing import preprocessing_functions

data_to_send = {
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


df = pd.DataFrame(data_to_send, index=[0])


df = preprocessing_functions(df=df)

feature_names_from_model = get_feature_names_from_model(experiment=experiment,
                                                        feature_names_directory_path=artifact_path)

# missing columns levels train and test.
for feature in feature_names_from_model:
    if feature not in df.columns:
        df[feature] = 0

prediction = model.predict(df[feature_names_from_model])[0]
logger.info(f"Prediction: {prediction}")