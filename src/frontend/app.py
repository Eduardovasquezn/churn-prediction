import os
from src.common.inference_utils import add_missing_features
from src.common.data_preprocessing import preprocessing_functions
from src.common.paths import artifacts_path
from src.common.logger import get_console_logger
from src.common.model_registry import (
    load_model_from_model_registry,
    get_feature_names_from_model,
)

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

logger = get_console_logger()


# Cache setup
@st.cache_data
def setup_app():
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

    return model, feature_names_from_model


model, feature_names_from_model = setup_app()


def main():
    st.title("Churn Prediction App")

    st.write("Enter customer data to predict churn:")

    # Define default values for input fields
    default_values = {
        "gender": ["Male", "Female"],
        "seniorcitizen": [0, 1],
        "dependents": ["No", "Yes"],
        "partner": ["No", "Yes"],
        "contract": ["Month-to-month", "One year", "Two year"],
        "tenure": [57],
        "paymentmethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "paperlessbilling": ["Yes", "No"],
        "monthlycharges": [53],
        "totalcharges": [3196],
        "deviceprotection": ["Yes", "No", "No internet service"],
        "onlinebackup": ["Yes", "No", "No internet service"],
        "onlinesecurity": ["Yes", "No", "No internet service"],
        "internetservice": ["DSL", "Fiber optic", "No"],
        "multiplelines": ["Yes", "No", "No phone service"],
        "phoneservice": ["Yes", "No"],
        "techsupport": ["Yes", "No", "No internet service"],
        "streamingmovies": ["Yes", "No", "No internet service"],
        "streamingtv": ["Yes", "No", "No internet service"],
    }

    # Create input fields in the sidebar with dropdowns for categorical features
    customer_data = {}
    for key, options in default_values.items():
        if key in ["tenure", "monthlycharges", "totalcharges"]:
            customer_data[key] = st.sidebar.slider(
                key, 0, 10000, default_values[key][0]
            )
        else:
            customer_data[key] = st.sidebar.selectbox(key, options)

    if st.button("Predict"):
        with st.spinner("Processing..."):

            logger.info("Transform to df...")
            df = pd.DataFrame(customer_data, index=[0])

            logger.info("Preprocessing...")
            df = preprocessing_functions(df=df)

            # missing columns levels train and test.
            add_missing_features(
                df=df, feature_names_from_model=feature_names_from_model
            )

            logger.info("Calculate predictions")
            prediction = model.predict(df[feature_names_from_model])[0]
            prediction_formatted = "Churn 😢" if prediction == 1 else "Not Churn 😊"

            st.write(
                f"<span style='font-size:25px'>{prediction_formatted}</span>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
