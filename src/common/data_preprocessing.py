import os
import pickle

import numpy as np
import pandas as pd
from comet_ml import Experiment, Artifact
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.common.logger import get_console_logger
from src.common.paths import artifacts_path

logger = get_console_logger()


def features_to_preprocess():
    logger.info("List of features to preprocess...")

    numerical_features = ["tenure", "monthlycharges", "totalcharges"]

    categorical_features = [
        "multiplelines", "internetservice", "onlinesecurity", "onlinebackup",
        "deviceprotection", "techsupport", "streamingmovies", "streamingtv",
        "phoneservice", "paperlessbilling", "contract", "paymentmethod", "gender",
        "dependents", "partner"
    ]

    return numerical_features, categorical_features


def preprocessing_functions(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Remove features")
    df.drop(columns=["customerid", "datetime_x", "datetime_y"], axis=1, inplace=True)

    logger.info("List of categorical_features features to transform...")
    numerical_features, categorical_features = features_to_preprocess()

    # Apply transformation functions to categorical features
    df = pd.get_dummies(df, columns=categorical_features, dtype=np.int64)

    return df

