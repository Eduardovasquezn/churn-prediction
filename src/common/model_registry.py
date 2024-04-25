import os
import pickle

import pandas as pd
from comet_ml import API, Experiment, Artifact
from sklearn.base import BaseEstimator

from src.common.logger import get_console_logger
from src.common.paths import model_path, artifacts_path

import time
from dotenv import load_dotenv
load_dotenv()

comet_ml_model_name = os.getenv('COMET_ML_MODEL_NAME')
comet_ml_api_key = os.getenv('COMET_ML_API_KEY')
logger = get_console_logger()



def save_model_to_model_registry(model: BaseEstimator, version: str, tag: str, status: str, experiment: Experiment,
                                 comet_ml_api_key: str, comet_ml_workspace: str) -> None:
    """
    Register the best model to the Comet Model Registry.

    Args:
    best_model: The best trained model.
    experiment: Comet ML Experiment object for logging.
    comet_api_key (str): Your Comet ML API key.

    Returns:
    None
    """
    logger.info('Saving model to disk')
    models_path = model_path()
    champion_model_path = os.path.join(models_path, 'model.pkl')

    with open(champion_model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Log model to Comet ML")
    experiment.log_model(comet_ml_model_name, champion_model_path)

    logger.info("Register model in model registry")
    experiment.register_model(comet_ml_model_name)

    set_model_status_and_tags(comet_ml_api_key=comet_ml_api_key, comet_ml_workspace=comet_ml_workspace,
                              version=version, status=status, tag=tag)

    experiment.end()


def set_model_status_and_tags(comet_ml_api_key: str, comet_ml_workspace: str,
                              version: str, status: str, tag: str) -> None:
    """
    Update model status and add tags in the Comet Model Registry.

    Args:
    comet_ml_api_key (str): Your Comet ML API key.
    comet_ml_workspace (str): Your Comet ML workspace name.
    version (str): Version of the model.
    status (str): Status to set for the model (e.g., "Production").
    tag (str): Tag to add to the model version.

    Returns:
    None
    """
    # Sleep for 5 seconds
    logger.info("Sleep for 10 seconds")
    time.sleep(10)

    logger.info("Connect to Comet ML API")
    api = API(api_key=comet_ml_api_key)
    logger.info("Get model from Registry")
    model = api.get_model(workspace=comet_ml_workspace, model_name=comet_ml_model_name)
    logger.info("Set status")
    # TODO: if status already assigned, skip it
    model.set_status(version=version, status=status)
    logger.info("Add tags to the model")
    model.add_tag(version=version, tag=tag)


def load_model_from_model_registry(
        workspace: str,
        api_key: str,
        model_name: str,
        status: str = 'Production',
):
    """Loads the production model from the remote model registry"""

    logger.info("Conect to Comet ML API")
    api = API(api_key=api_key)

    logger.info("Get model details")
    model_details = api.get_registry_model_details(workspace, model_name)['versions']
    model_versions = [md['version'] for md in model_details if md['status'] == status]

    if len(model_versions) == 0:
        logger.error('No production model found')
        raise ValueError('No production model found')
    else:
        logger.info(f'Found {status} model versions: {model_versions}')
        model_version = model_versions[0]

    logger.info("Specify model from Comet ML")
    model = api.get_model(workspace=workspace, model_name=model_name)

    logger.info("Download model from Comet ML")
    model_directory_path = model_path()
    model.download(version=model_version,
                   output_folder=model_directory_path,
                   expand=True)

    champion_model_path = os.path.join(model_directory_path, 'model.pkl')
    logger.info("Load model locally")
    with open(champion_model_path, "rb") as f:
        model = pickle.load(f)

    return model


def save_feature_names_from_model(x_train: pd.DataFrame, experiment: Experiment) -> None:
    # Path of the artifact
    artifact_path = artifacts_path()
    feature_names_path = os.path.join(artifact_path, 'feature_names_from_model.pkl')

    x_train_columns = list(x_train.columns)

    logger.info(f'Saving feature names from the model in local path...')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(x_train_columns, f)

    logger.info("Create artifact with the feature names of the model")
    artifact = Artifact(name="feature_names_model", artifact_type="feature_names")
    logger.info("Add feature names to artifact")
    artifact.add(feature_names_path)
    logger.info("Log feature names")
    experiment.log_artifact(artifact)

#TODO: create a function that reads from an existing experiment rather than creating a new one
def get_feature_names_from_model(experiment: Experiment, feature_names_directory_path: str,
                                 artifact_name: str = "feature_names_model", api_key: str = comet_ml_api_key):
    """
    Load the feature names artifact from Comet ML and save it locally.

    Args:
    experiment: Comet ML Experiment object.
    scaler_path (str): Path to save the feature names locally.
    artifact_name (str): Name of the artifact.

    Returns:
    None
    """
    logger.info("Get feature names artifact")
    get_scaler = experiment.get_artifact(artifact_name)
    logger.info("Download feature names artifact")
    get_scaler.download(path=feature_names_directory_path, overwrite_strategy=True)

    feature_names_path = os.path.join(feature_names_directory_path, 'feature_names_from_model.pkl')

    logger.info("Load feature names artifact")
    with open(feature_names_path, 'rb') as f:
        feature_names_from_model = pickle.load(f)

    # Workaround for deleting the dummy experiment created during the download
    api = API(api_key=api_key)
    api.delete_experiment(experiment_key=experiment.get_key())

    return feature_names_from_model