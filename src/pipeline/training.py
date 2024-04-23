import sys
import os

# Add the directory containing the src package to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from comet_ml import Experiment

from src.common.data import load_and_split_data
from src.common.data_preprocessing import preprocessing_functions
from src.common.logger import get_console_logger
from src.common.model_registry import save_model_to_model_registry, save_feature_names_from_model
from src.common.modelling import evaluate_and_log_predictions, \
    optimize_optuna, fit_best_model
from src.common.save_data import save_training_data

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()

comet_ml_api_key = os.getenv("COMET_ML_API_KEY", "")
comet_ml_workspace = os.getenv("COMET_ML_WORKSPACE", "")
comet_ml_project_name = os.getenv("COMET_ML_PROJECT_NAME", "")


def run_training_pipeline(model_type, version, tag, status):

    # Train-test split
    x_train, x_test, y_train, y_test = load_and_split_data()

    logger.info("Specify Comet ML experiment")
    experiment = Experiment(
        api_key=comet_ml_api_key,
        workspace=comet_ml_workspace,
        project_name=comet_ml_project_name
    )

    logger.info("Saving training data in temp path and uploading artifact...")
    save_training_data(x_train=x_train, y_train=y_train, experiment=experiment)

    x_train = preprocessing_functions(df=x_train)

    logger.info("Save features' names of the model to fit")
    save_feature_names_from_model(x_train=x_train, experiment=experiment)

    logger.info("Training the model")
    best_hyperparameters = optimize_optuna(n_trials=10, model_type=model_type, x_train=x_train, y_train=y_train,
                                  experiment=experiment)

    logger.info(f"best_params: {best_hyperparameters}")

    logger.info("Fit model with best hyperparameters")
    model = fit_best_model(model_type=model_type, x_train=x_train, y_train=y_train, **best_hyperparameters)

    evaluate_and_log_predictions(best_estimator=model, x_test=x_test, y_test=y_test, experiment=experiment)

    save_model_to_model_registry(model=model, version=version, tag=tag, status=status, experiment=experiment,
                                 comet_ml_api_key=comet_ml_api_key, comet_ml_workspace=comet_ml_workspace)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--version', type=str, default='1.0.0')
    parser.add_argument('--tag', type=str, default='random_forest')
    parser.add_argument('--status', type=str, default='Staging')
    args = parser.parse_args()

    logger.info('Training model')
    run_training_pipeline(model_type=args.model_type, version=args.version, tag=args.tag, status=args.status)
