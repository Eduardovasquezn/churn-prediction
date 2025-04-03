import fire

from src.common import (
    merge_dfs,
    extract_data,
    convert_features_to_lowercase,
    transform_features,
)
from src.common import get_console_logger
from src.common import save_df

logger = get_console_logger()


def run_feature_pipeline():

    # Extract data
    demography_df, customer_info_df, subscriptions_df = extract_data()

    # List of df
    dataframes = [demography_df, customer_info_df, subscriptions_df]

    # Convert feature names to lowercase
    [convert_features_to_lowercase(df) for df in dataframes]

    # Transform features
    customer_info_df = transform_features(df=customer_info_df)

    logger.info("Merge dataframes based on common identifier...")
    merged_dataframes = merge_dfs(dataframes, on="customerid")

    # Save df
    save_df(merged_dataframes)


if __name__ == "__main__":
    fire.Fire(run_feature_pipeline)
