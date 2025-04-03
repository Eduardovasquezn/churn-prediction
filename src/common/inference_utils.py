import os

import pandas as pd
from pydantic import BaseModel

from src.common import get_console_logger
from src.common import load_model_from_model_registry, get_feature_names_from_model
from src.common import artifacts_path

logger = get_console_logger()


class CustomerData(BaseModel):
    customerid: str
    gender: str
    seniorcitizen: int
    dependents: str
    partner: str
    contract: str
    tenure: int
    paymentmethod: str
    paperlessbilling: str
    monthlycharges: float
    totalcharges: float
    datetime_x: str
    deviceprotection: str
    onlinebackup: str
    onlinesecurity: str
    internetservice: str
    multiplelines: str
    phoneservice: str
    techsupport: str
    streamingmovies: str
    streamingtv: str
    datetime_y: str


def add_missing_features(df: pd.DataFrame, feature_names_from_model: list) -> None:
    """
    Adds missing features to the DataFrame with default value 0.

    Parameters:
        df (pd.DataFrame): The DataFrame to which missing features will be added.
        feature_names_from_model (list): List of feature names from the model.

    """
    logger.info("Adding missing features that were used to fit the model...")
    for feature in feature_names_from_model:
        if feature not in df.columns:
            df[feature] = 0


def setup_model():
    comet_ml_api_key = os.getenv("COMET_ML_API_KEY", "")
    comet_ml_workspace = os.getenv("COMET_ML_WORKSPACE", "")
    comet_ml_model_name = os.getenv("COMET_ML_MODEL_NAME", "")
    comet_ml_project_name = os.getenv("COMET_ML_PROJECT_NAME", "")

    from comet_ml import Experiment

    experiment = Experiment(
        api_key=comet_ml_api_key,
        workspace=comet_ml_workspace,
        project_name=comet_ml_project_name,
    )

    model = load_model_from_model_registry(
        workspace=comet_ml_workspace,
        api_key=comet_ml_api_key,
        model_name=comet_ml_model_name,
    )

    artifact_path = artifacts_path()
    feature_names_from_model = get_feature_names_from_model(
        experiment=experiment, feature_names_directory_path=artifact_path
    )
    experiment.end()
    return model, feature_names_from_model
