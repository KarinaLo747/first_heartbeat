import os
import pandas
import logging


# Setup logger
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter: logging.Formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

# Define handlers
fh: logging.FileHandler = logging.FileHandler(f'{__name__}.log')
fh.setFormatter(formatter)
sh: logging.StreamHandler = logging.StreamHandler()

# Add handlers to logger
logger.addHandler(fh)
logger.addHandler(sh)


def create_output_dir(data_dir: str, old_subdir: str = 'raw', new_subdir: str = 'processed') -> str:
    """Creates an output directory if it does not exist already with the same experiment subdirectory names as the input data directory.
    Default settings assume the Cookiecutter Data Science project template is used.

    Args:
        data_dir (str): Input data dirpath.
        old_subdir (str, optional): Subdir name to be changed. Defaults to 'raw'.
        new_subdir (str, optional): Subdir name to be changed to. Defaults to 'processed'.

    Returns:
        str: Dirpath of the newly created dirpath.

    Example:
        >>> output_dir = create_output_dir('data/raw/experiment1/')
        >>> print(output_dir)
        'data/processed/experiment1/'
    """

    output_dir: str = data_dir.replace(old_subdir, new_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f'Created {output_dir}')
    else:
        logger.info(f'{output_dir} already exists and has not been overwritten')

    return output_dir


def norm_df(df: pandas.DataFrame) -> pandas.DataFrame:
    """Min max normalise input DataFrame column wise to values between 0 and 1.

    Formula:
    
        x_norm = (x - x_min) / (x_max - x_min)

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Normalised DataFrame.
    """

    norm_data: pandas.DataFrame = (df-df.min())/(df.max()-df.min())
    logger.info('Min max normalised DataFrame')

    return norm_data
