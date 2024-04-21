import os

def create_directory_if_not_exists(directory_path: str) -> str:
    """
    Create a directory if it doesn't exist.

    Args:
    directory_path (str): Path of the directory to create.

    Returns:
    str: Path of the created directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def data_path():
    # Define the directory path
    directory_path = '../../workspace/data/'

    # Create the directory if it doesn't exist
    create_directory_if_not_exists(directory_path)

    return directory_path


def model_path():
    # Define the directory path
    directory_path = '../../workspace/models/'

    # Create the directory if it doesn't exist
    create_directory_if_not_exists(directory_path)

    return directory_path

def artifacts_path():
    # Define the directory path
    directory_path = '../../workspace/artifacts/'

    # Create the directory if it doesn't exist
    create_directory_if_not_exists(directory_path)

    return directory_path
