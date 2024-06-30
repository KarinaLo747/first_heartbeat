import pandas
import numpy
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from first_heartbeat.load_data import load_circles
from first_heartbeat.utils import create_output_dir, norm_df, get_exp_info, real_time
from first_heartbeat.constants import circle_roi_cols
from first_heartbeat.plotter import time_vs_fluoresence


def peak_find(y_data: any, prominence: float = 0.5, height: float = None, width: float = None) -> numpy.ndarray:
    """Returns the index of peaks found from an array-like input. Uses the scipy.signal.find_peaks.

    Args:
        y_data (any): Array-like input containing possible peaks to be found, such as fluoresence intensity data containing signals.
        prominence (float, optional): As described by scipy.signal.find_peaks documentation. Defaults to 0.5.
        height (float, optional): As described by scipy.signal.find_peaks documentation. Defaults to None.
        width (float, optional): As described by scipy.signal.find_peaks documentation. Defaults to None.

    Returns:
        numpy.ndarray: Index of peaks found.
    """

    peaks, _ = find_peaks(
        x=y_data,
        prominence=prominence,
        height=height,
        width=width,
    )

    return peaks


def run_analysis(data_dir: str, filter_regex: str = None) -> None:

    # Define name of dir for all outputs
    output_dir = create_output_dir(data_dir=data_dir)

    csv_stem, data = load_circles(csv_dir=data_dir, filter_regex=filter_regex)
    exp_info = get_exp_info(csv_stem=csv_stem)
    sec_per_frame = exp_info['sec_per_frame']

    norm_data = norm_df(data)

    time_vs_fluoresence(
        data=norm_data,
        sec_per_frame=sec_per_frame,
        output_dir=output_dir,
        title='All ROI',
    )

    sides: dict[str: list['str']] = {
        'Left side': [circle_roi_cols[col] for col in ['mean_LL', 'mean_LI', 'mean_LM']],
        'Right side': [circle_roi_cols[col] for col in ['mean_RL', 'mean_RI', 'mean_RM']],
    }
    
    for title, side in sides.items():
        time_vs_fluoresence(
            data=norm_data[side],
            sec_per_frame=sec_per_frame,
            output_dir=output_dir,
            title=title,
        )
        time_vs_fluoresence(
            data=norm_data[side],
            sec_per_frame=sec_per_frame,
            output_dir=output_dir,
            title=title,
            xlim=[0, 10],
        )

    y_data: pandas.Series = norm_data[circle_roi_cols['mean_LL']]
    peaks_ind: numpy.ndarray = peak_find(
        y_data=y_data,
        prominence=0.5,
    )
    print(f'{peaks_ind = }')

    # calc_beat_freq()

    # get_upbeat()

    # calc_t_half()

    # calc_direction()

    # calc_phase_diff()

    # to_csv()
