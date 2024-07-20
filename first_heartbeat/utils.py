import os
import pandas
import glob
import pathlib
import datetime
import logging
import numpy
import numpy as np
from dataclasses import dataclass
from first_heartbeat.constants import decimal_places


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
# logger.addHandler(sh)


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


@dataclass
class Embryo:
    date: datetime.date
    mouse_line: str
    dpc: float
    exp: int
    embryo: int
    mag: int
    total_frames: int
    cut: str
    section: str
    repeat: int
    linestep: int
    stage: int
    duration: float
    sec_per_frame: float


def get_exp_info(csv_stem: str) -> dict[str: any]:
    """Extracting information embedded in the CSV file name. The seconds per frame is calculated and returned in the output dictionary with key 'sec_per_frame'.

    The CSV file naming convention used is as follows with key information is delimited by \'_\':
        {yymmdd}_{mouse_line}_E{dpc}_Exp{exp}_E{embryo}_{mag}x_{total_frames}cyc_{cut}_{section}_t{repeat}_ls{linestep}_stage-{stage}_dur-{duration}.csv

        {yymmdd}                    Date
        {mouse_line}                Mouse line
        E{dpc}                      Days past coitus
        Exp{exp}                    Emperiment number
        E{embryo}                   Embryo number
        {mag}x                      Microscope magnification
        {total_frames}cyc           Total number of frames
        {cut}                       Cut: precut or postcut
        {section}                   Section cut
        t{repeat}                   Repeat number
        ls{linestep}                Linestep amount
        stage-{stage}               (User defined) Stage of embryo development
        dur-{duration}              (User defined) Total duration of scanning time in seconds

    Important:
        The stage and duration is dependent on two hard-coded annotations by the user as follows:
        {as_saved_by_the_microscope}_stage-{stage_number}_dur-{duration_as_float}.csv

    Args:
        csv_stem (str): Stem is the filename without the dirpath and extension.

    Returns:
        dict[str, any]: Keys: ('date', 'mouse_line', 'dpc', 'exp', 'embryo', 'mag', 'total_frames', 'cut', 'section', 'repeat', 'linestep', 'stage', 'duration', 'sec_per_frame')
    """


    # Split CSV stem. Stem is the filename without the dirpath and extension
    info_lst: list[str] = csv_stem.split('_')

    # Date loaded as a datetime.date
    sdate: str = info_lst[0]  # short-hand date
    yyyy: int = int('20' + sdate[:2])
    mm: int = int(sdate[2:4])
    dd: int = int(sdate[4:])
    date: datetime.date = datetime.date(yyyy, mm, dd)

    # Mouse line
    mouse_line: str = info_lst[1]

    # Days post coitus (dpc)
    dpc: float = float(info_lst[2][1:])

    # Experiment number
    exp: int = int(info_lst[3][3:])

    # Embryo number from litter
    embryo: int = int(info_lst[4][1:])

    # Magnification of microscope
    mag: int = int(info_lst[5][:-1])

    # Total number of frames scanned
    total_frames: int = int(info_lst[6][:-3])

    # Precut or postcut
    cut: str = info_lst[7]

    # Section the embryo was cut at
    # Indexing not used in case non-abbrevated words are used such as 'upper'
    section: str = info_lst[8].replace('.', '')

    # Experiment repeat number
    repeat: int = int(info_lst[9][1:])

    # Microscope linestep amount
    linestep: int = int(info_lst[10][2:])

    # Annotated stage of heart development
    stage: int = int(info_lst[11].split('-')[1])

    # Annotated total duration of scanning time in seconds
    duration: float = float(info_lst[12].split('-')[1])  # duration of scan

    # Calculated seconds per frame
    sec_per_frame: float = duration / total_frames

    # Save all the experiment info to the dataclass Embryo
    embryo:Embryo = Embryo(
        date=date,
        mouse_line=mouse_line,
        dpc=dpc,
        exp=exp,
        embryo=embryo,
        mag=mag,
        total_frames=total_frames,
        cut=cut,
        section=section,
        repeat=repeat,
        linestep=linestep,
        stage=stage,
        duration=duration,
        sec_per_frame=sec_per_frame,
    )

    return embryo


def real_time(x_data: numpy.ndarray, sec_per_frame: float) -> numpy.ndarray:
    """Multiply an array of frame numbers by a scaler, such as the calculated seconds per frame.
    Broadcasting the scaling of a numpy.ndarray is used. The the data type of the input is checked
    and converted to a numpy.ndarray if not already.

    Args:
        x_data (numpy.ndarray): Input data, such as an array of frame numbers.
        sec_per_frame (float): A scaler amount.

    Returns:
        numpy.ndarray: Output array in seconds.
    """

    if not isinstance(x_data, numpy.ndarray):
        x_np = np.array(x_data)
    else:
        x_np = x_data

    x_sec: numpy.ndarray = x_np*sec_per_frame

    return x_sec


def calc_beat_freq(duration: float, num_peaks: int) -> float:
    """Calculates beat frequency by dividing the duration of the scan by the number of peaks found.
    The output is rounded to a pre-defined number of decimal places as defined in first_heartbeat.constants.

    Args:
        duration (float): Duration of scan in seconds.
        num_peaks (int): Number of peaks found.

    Returns:
        float: The beat frequency in units of per second (Hz).
    """

    beat_freq: float = duration / num_peaks
    beat_freq: float = round(beat_freq, decimal_places)

    return beat_freq
