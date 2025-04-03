import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import get_console_logger
from src.common import data_path

logger = get_console_logger()


def extract_data() -> pd.DataFrame:
    """
    Extract demography data, customer data, and subscription data from URLs.

    Returns:
    - tuple: Tuple containing demography_df, customer_info_df, and subscriptions_df DataFrames.
    """

    logger.info("Fetching demography data...")
    demography_df = pd.read_csv(
        "https://repo.hops.works/dev/davit/churn/demography.csv"
    )

    logger.info("Fetching customer data...")
    customer_info_df = pd.read_csv(
        "https://repo.hops.works/dev/davit/churn/customer_info.csv",
        parse_dates=["datetime"],
    )

    logger.info("Fetching subscription data...")
    subscriptions_df = pd.read_csv(
        "https://repo.hops.works/dev/davit/churn/subscriptions.csv",
        parse_dates=["datetime"],
    )

    return demography_df, customer_info_df, subscriptions_df


def convert_columns_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert specified columns to numeric, treating errors as NaN.

    Args:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to convert to numeric.

    Returns:
    pd.DataFrame: DataFrame with specified columns converted to numeric.
    """
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    return df


def replace_nan_values(df: pd.DataFrame, columns: list, value: int = 0) -> pd.DataFrame:
    """
    Replace NaN values in specified columns with a specified value.

    Args:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names in which to replace NaN values.
    value (int, float): Value to replace NaN values with. Default is 0.

    Returns:
    pd.DataFrame: DataFrame with NaN values in specified columns replaced.
    """
    df[columns] = df[columns].fillna(value)
    return df


def replace_categorical_values(
    df: pd.DataFrame, columns: list, mapping: dict
) -> pd.DataFrame:
    """
    Replace categorical values in specified columns with specified mapping.

    Args:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names in which to replace categorical values.
    mapping (dict): Dictionary mapping original values to replacement values.

    Returns:
    pd.DataFrame: DataFrame with categorical values in specified columns replaced.
    """
    df[columns] = df[columns].replace(mapping)
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Convert the totalcharges column to numeric, treating errors as NaN")
    df = convert_columns_to_numeric(df, columns=["totalcharges"])

    logger.info("Replace NaN values in the totalcharges column with 0")
    df = replace_nan_values(df, columns=["totalcharges"])

    logger.info("Replace values in the churn column with 0 for No and 1 for Yes")
    df = replace_categorical_values(df, columns=["churn"], mapping={"No": 0, "Yes": 1})

    return df


def merge_dfs(dataframes: list, on: str) -> pd.DataFrame:
    merged_df = dataframes[0]

    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=on)

    return merged_df


def convert_features_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform all features (columns) in a DataFrame to lowercase.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with all features transformed to lowercase.
    """
    logger.info("Transform features' names to lowercase")
    df.columns = map(str.lower, df.columns)
    return df


def load_and_split_data(
    parquet_file_path: str = "merged_df.parquet",
    target_column: str = "churn",
    test_size: float = 0.2,
    random_state: int = 10,
):
    """
    Load data from a Parquet file, perform train-test split, and return the split datasets.

    Args:
    parquet_file_path (str): Path to the Parquet file containing the dataset.
    target_column (str): Name of the target column.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    random_state (int): Controls the randomness of the dataset splitting (default is 10).

    Returns:
    tuple: A tuple containing four DataFrames: x_train, x_test, y_train, and y_test.
    """
    logger.info("Loading data")
    data_directory_path = data_path()
    full_data_path = os.path.join(data_directory_path, parquet_file_path)
    # Load data from Parquet file
    df = pd.read_parquet(full_data_path)

    logger.info("Splitting data")
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1),
        df[target_column],
        test_size=test_size,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test
