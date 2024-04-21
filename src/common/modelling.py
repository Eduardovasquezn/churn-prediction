import numpy as np
import optuna
import pandas as pd
from comet_ml import Experiment
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from typing import Callable

from src.common.data_preprocessing import preprocessing_functions
from src.common.logger import get_console_logger

logger = get_console_logger()


def get_model(model_type, **kwargs) -> Callable:
    """
    Get a classifier model based on the specified model type.

    Args:
    model_type (str): Type of classifier model. Options: 'XGB', 'RandomForest', 'LogisticRegression', 'SVM'.
    **kwargs: Additional keyword arguments specific to the chosen model.

    Returns:
    model: Initialized classifier model.
    """
    logger.info("Load model")
    if model_type == 'xgboost':
        model = XGBClassifier(random_state=10, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight="balanced_subsample", random_state=10, **kwargs)
    elif model_type == 'lgbm':
        model = LGBMClassifier(random_state=10, **kwargs)
    else:
        raise ValueError("Invalid model_type. Choose from 'xgboost', 'random_forest', 'lgbm'.")

    return model

def get_hyperparameters(model_type, trial):
    """
    Get the hyperparameters of classifier model based on the specified model type.

    Args:
    model_type (str): Type of classifier model. Options: 'XGB', 'RandomForest', 'LogisticRegression', 'SVM'.
    **kwargs: Additional keyword arguments specific to the chosen model.

    Returns:
    hyperparameters: Hyperparameters of classifier model.
    """
    logger.info("Load hyperparameters")
    if model_type == 'xgboost':
        hyperparameters = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 6)
        }
    elif model_type == 'random_forest':
        hyperparameters = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 9),
            'max_features': trial.suggest_categorical("max_features", ['auto', 'sqrt'])
        }
    elif model_type == 'lgbm':
        hyperparameters = {
            'num_leaves': trial.suggest_int('num_leaves', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 6)
        }

    return hyperparameters

def evaluate_and_log_predictions(best_estimator, x_test, y_test, experiment):
    """
    Calculate predictions on the test set, compute ROC score, and upload confusion matrix to Comet ML.

    Args:
    best_estimator: The best estimator model.
    x_test: Test features.
    y_test: Test labels.
    experiment: Comet ML Experiment object for logging.

    Returns:
    None
    """
    logger.info("Preprocessing of x_test")
    x_test = preprocessing_functions(df=x_test)

    logger.info("Calculate predictions on test set")
    y_pred = best_estimator.predict(x_test)

    roc_score = roc_auc_score(y_test, y_pred)
    logger.info(f"ROC score on testing set: {roc_score}")
    experiment.log_metric('ROC - test set', roc_score)

    logger.info("Upload confusion matrix to Comet ML")
    experiment.log_confusion_matrix(y_test.tolist(), y_pred.tolist())


def optimize_optuna(n_trials: int, model_type: str, x_train: pd.DataFrame, y_train: pd.DataFrame,
              experiment: Experiment):


    def objective(trial):

        # Load hyperparameters
        hyperparameters = get_hyperparameters(model_type=model_type, trial=trial)

        # Specify model with respective hyperparameters
        model = get_model(model_type=model_type, **hyperparameters)

        # Use KFold cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=10)
        roc_auc_scores = []
        f1_scores = []
        precision_scores = []
        for train_index, val_index in kf.split(x_train):
            x_train_set, x_val_set = x_train.iloc[train_index], x_train.iloc[val_index]
            y_train_set, y_val_set = y_train.iloc[train_index], y_train.iloc[val_index]

            logger.info("Fit model")
            model.fit(x_train_set, y_train_set)

            logger.info("Compute predictions")
            y_pred = model.predict(x_val_set)

            logger.info("Evaluate model")
            roc_score = roc_auc_score(y_val_set, y_pred)
            roc_auc_scores.append(roc_score)

            f1 = f1_score(y_val_set, y_pred)
            f1_scores.append(f1)

            precision = precision_score(y_val_set, y_pred)
            precision_scores.append(precision)

        logger.info("Metrics:")
        roc_score_mean = np.array(roc_auc_scores).mean()
        logger.info(f"ROC score: {roc_score_mean}")
        experiment.log_metric('ROC - cross validation', roc_score_mean)

        f1_score_mean = np.array(f1_scores).mean()
        logger.info(f"F1 score: {f1_score_mean}")
        experiment.log_metric('F1 - cross validation', f1_score_mean)

        precision_score_mean = np.array(precision_scores).mean()
        logger.info(f"Precision score: {precision_score_mean}")
        experiment.log_metric('Precision - cross validation', precision_score_mean)

        # Return the mean score
        return roc_score_mean


    logger.info('Start hyperparameters search...')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Get the best hyperparameters and results
    best_params = study.best_params
    best_auc = study.best_value  # Convert back to positive accuracy

    logger.info(f"Best AUC: {best_auc}")
    logger.info(f"Best best_hyperparameters: {best_params}")

    experiment.log_metric('Best ROC - cross validation', best_auc)
    experiment.log_parameters(best_params)

    return best_params

def fit_best_model(model_type: str, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs) -> Callable:
    # Specify model with respective hyperparameters
    model = get_model(model_type=model_type, **kwargs)
    logger.info(f"Fitting model: {model_type}...")
    model.fit(x_train, y_train)

    return model