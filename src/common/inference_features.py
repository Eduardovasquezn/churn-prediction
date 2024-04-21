import pandas as pd
from pydantic import BaseModel

from src.common.logger import get_console_logger

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
