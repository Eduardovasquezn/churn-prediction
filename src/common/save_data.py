import os
import tempfile

import pandas as pd
from comet_ml import Artifact, Experiment

from src.common.logger import get_console_logger
from src.common.paths import data_path

logger = get_console_logger()


def save_df(df: pd.DataFrame) -> None:
    """
    Save merged DataFrame in Parquet format.

    Args:
    - merged_df (pandas.DataFrame): DataFrame to be saved.

    Returns:
    - str: Path where the DataFrame is saved.
    """
    logger.info("Path to save merged df")
    data_directory_path = data_path()

    logger.info("Save merged dataframe")
    df.to_parquet(os.path.join(data_directory_path, 'merged_df.parquet'))

# Reuse data produced in their experimentation pipeline, and allow it to be tracked, versioned, consumed, and analyzed
# in a managed way.
# Iterate on their datasets over time, track which model used which version of the dataset, and schedule
# model re-training.
def save_training_data(x_train: pd.DataFrame, y_train: pd.DataFrame, experiment: Experiment) -> None:
    """
    Save training data to a Parquet file and log it as an artifact to the Comet.ml experiment.

    Args:
    x_train (pd.DataFrame): DataFrame containing the features of the training data.
    y_train (pd.DataFrame): DataFrame containing the labels of the training data.
    experiment (comet_ml.Experiment): Comet.ml Experiment object for logging.

    Returns:
    None
    """
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the temporary output path
        output_path = os.path.join(tmpdirname, "training_data.parquet")

        logger.info("Create df with training data")
        training_data = pd.concat([x_train, y_train], axis=1)

        logger.info("Save training data in a temporary path...")
        training_data.to_parquet(output_path)

        # Create and add the artifact
        logger.info("Create dataset artifcact")
        artifact = Artifact(name="training_churn_data", artifact_type="dataset")
        logger.info("Add artifact")
        artifact.add(output_path)
        logger.info("Log artifact")
        experiment.log_artifact(artifact)
