import logging
import pandas
import pandas as pd


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

# Define handlers
fh = logging.FileHandler(f'{__name__}.log')
fh.setFormatter(formatter)
sh = logging.StreamHandler()

# Add handlers to logger
logger.addHandler(fh)
logger.addHandler(sh)


def load_circles(csv_path: str, filter_regex: str = None) -> pandas.DataFrame:
    """Load CSV file using pandas with the option to filter by columns using column name string regex.
    Assumed ImageJ > ROI manager > More > Multimeasure was used to create input CSV file, therefore 0th column is frame number.

    Args:
        csv_path (str): Input CSV filepath.
        filter_regex (str, optional): Filter column headers by this string regex. Defaults to None.

    Raises:
        ValueError: If filter using input string regex returns an empty DataFrame.

    Returns:
        pandas.DataFrame: ImageJ circles multimeasure output loaded as a DataFrame.
    """

    data = pd.read_csv(csv_path, index_col=0)
    data.index.name = 'frame_num'
    logger.info(f'Loaded {csv_path}')

    if filter_regex:
        data = data.filter(regex=filter_regex)
        if data.empty:
            raise ValueError(f'Filter using regex \'{filter_regex}\' returned an empty DataFrame. Try again.')
        logger.info(f'Filtered columns using regex \'{filter_regex}\': {', '.join(data.columns)}')

    return data
