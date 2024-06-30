import logging
import pandas
import pandas as pd
import glob
import pathlib


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


def load_circles(csv_dir: str, filter_regex: str = None) -> tuple[str, pandas.DataFrame]:
    """Load CSV file using pandas with the option to filter by columns using column name string regex.
    Assumed ImageJ > ROI manager > More > Multimeasure was used to create input CSV file, therefore 0th column is frame number.

    Args:
        csv_dir (str): Directory path containing input CSV file. Only one CSV file can be present.
        filter_regex (str, optional): Filter column headers by this string regex. Defaults to None.

    Raises:
        ValueError: If filter using input string regex returns an empty DataFrame.
        ValueError: If there are too many CSV files in the input CSV directory. There must only be one.

    Returns:
        str: CSV file name with the dirpath removed.
        pandas.DataFrame: ImageJ circles multimeasure output loaded as a DataFrame.
    """

    # Search for CSV file
    csv_files: list = glob.glob(csv_dir+'*.csv')
    if len(csv_files) > 1:
        raise ValueError('There are too many CSV files in the input directory. There must only be one CSV file.')
    csv_file: str = csv_files[0]
    csv_name: str = pathlib.Path(csv_file).name

    # Load CSV using pandas
    data: pandas.DataFrame = pd.read_csv(csv_file, index_col=0)
    data.index.name = 'frame_num'
    logger.info(f'Loaded {csv_file}')

    # Filter DataFrame by column header
    if filter_regex:
        data: pandas.DataFrame = data.filter(regex=filter_regex)
        if data.empty:
            raise ValueError(f'Filter using regex \'{filter_regex}\' returned an empty DataFrame. Try again.')
        logger.info(f'Filtered columns using regex \'{filter_regex}\': {', '.join(data.columns)}')

    return csv_name, data
