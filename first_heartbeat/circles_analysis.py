import pandas
import numpy
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from first_heartbeat.load_data import load_circles
from first_heartbeat.utils import create_output_dir, norm_df, get_exp_info, real_time, calc_beat_freq
from first_heartbeat.constants import circle_roi_cols
from first_heartbeat.plotter import time_vs_fluoresence


def find_peak_ind(y_data: any, prominence: float = 0.5, height: float = None, width: float = None) -> numpy.ndarray:
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


def find_peak_base_ind(y_data: any, peaks_ind: numpy.ndarray, rel_height: float = 0.95) -> numpy.ndarray:
    """Returns the index of the left bases found from an array-like input. Uses the scipy.signal.peak_widths.

    Option to return right bases if the following code is added:
        >>> right_bases: numpy.ndarray = widths[3].astype(int)

    Args:
        y_data (any): Array-like input containing possible peaks to be found, such as fluoresence intensity data containing signals.
        peaks_ind (numpy.ndarray): Index of peaks found using scipy.signal.find_peaks on the same input y_data.
        rel_height (float, optional): As described by scipy.signal.peak_widths documentation.. Defaults to 0.95.

    Returns:
        numpy.ndarray: Index of left bases found.
    """

    # Find left and right bases of signals
    widths = peak_widths(
        y_data, peaks_ind,
        rel_height=rel_height,
    )

    # Left bases have index of 2
    # Turn float into int
    left_bases: numpy.ndarray = widths[2].astype(int)

    return left_bases


def find_upbeats(
    y_data: pandas.Series,
    left_bases_ind: numpy.ndarray,
    peaks_ind: numpy.ndarray,
    ) -> tuple[list[numpy.ndarray]]:
    """Returns a list of numpy.ndarray containing the x and y coordinates of the upbeat only for each signal.
    The upbeat is defined by the starting at the left base of a signal and ending at the peak of a signal.

    Args:
        y_data (pandas.Series): Fluoresence data containing signals.
        left_bases_ind (numpy.ndarray): Array containing the indexes of the signal left bases.
        peaks_ind (numpy.ndarray): Array containing the indexes of the signal peaks.

    Returns:
        tuple[list[numpy.ndarray]]: Two lists of numpy.ndarray containing the x and y coordinates of the signal upbeats.
    """

    # Extract x values from index and y from values.
    # Save as numpy arrays
    x_np = y_data.index.to_numpy()
    y_np = y_data.values

    x_upbeat_lst: list[int] = []
    y_upbeat_lst: list[float] = []

    # Loop over the indexes at which the left bases and peaks are found
    for start_ind, end_ind in zip(left_bases_ind, peaks_ind):

        # Define the x and y windows to calculate the t_half from
        # Slice by the left base and peak indexes to get the corresponding x and y values
        x_upbeat = x_np[start_ind:end_ind+1]
        y_upbeat = y_np[start_ind:end_ind+1]

        x_upbeat_lst.append(x_upbeat)
        y_upbeat_lst.append(y_upbeat)

    return x_upbeat_lst, y_upbeat_lst


def calc_t_half(
    x_upbeat_lst: list[numpy.ndarray],
    y_upbeat_lst: list[numpy.ndarray],
    kind: str = 'linear',
) -> tuple[numpy.ndarray]:
    """Returns the x and y cooridnates of the t_half of the pre-calculated upbeats of each signal.

    Process:
        1) Calculate the halfway point between the y coordinates for the left base and peak of a signal upbeat.
        2) Define an interpolation function with the signal upbeat x and y coordinates as training data. Note,
           the general formula is x = f(y) given we have y coordinates of the t_half from step 1 and we want to
           interpolate the x coordinate of the t_half.
        3) Return the x and y coordinates of t_half as seperate numpy.ndarray.

    Args:
        x_upbeat_lst (list[numpy.ndarray]): A list of numpy.ndarray for the x coordiantes of all the data points within a signal upbeat.
        y_upbeat_lst (list[numpy.ndarray]): A list of numpy.ndarray for the y coordiantes of all the data points within a signal upbeat.
        kind (str, optional): As described by scipy.interpolate.interp1d.. Defaults to 'linear'.

    Returns:
        tuple[numpy.ndarray]: Two numpy.ndarray containing the x and y coordinates of t_half of each signal.
    """

    # Empty lists for x and y coordinates for both t_halfs and upbeats
    x_t_half_np: numpy.ndarray = np.array([], dtype=float)
    y_t_half_np: numpy.ndarray = np.array([], dtype=float)

    # Loop over the indexes at which the left bases and peaks are found
    for x_upbeat, y_upbeat in zip(x_upbeat_lst, y_upbeat_lst):

        # Calculate the half way point of the y window
        y_t_half = (y_upbeat.max() + y_upbeat.min()) / 2

        # Define interpolation function trained on y and x windows
        # Such that frame number is a function of the fluoresence intensity
        # frame number = f(fluoresence intensity)
        f = interp1d(y_upbeat, x_upbeat, kind)
        x_t_half = f(y_t_half)

        # Append results
        x_t_half_np: numpy.ndarray = np.append(x_t_half_np, x_t_half)
        y_t_half_np: numpy.ndarray = np.append(y_t_half_np, y_t_half)

    return x_t_half_np, y_t_half_np


def run_analysis(
    data_dir: str,
    filter_regex: str = None,
    prominence: float = 0.5,
    rel_height: float = 0.95,
    kind: str = 'linear',
    ) -> None:

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

    x_t_half_dict = {}
    y_t_half_dict = {}

    for roi_name in ['mean_LL', 'mean_LM', 'mean_RL', 'mean_RM']:

        y_data: pandas.Series = norm_data[circle_roi_cols[roi_name]]
        peaks_ind: numpy.ndarray = find_peak_ind(y_data=y_data, prominence=prominence)
        left_bases_ind: numpy.ndarray = find_peak_base_ind(y_data=y_data, peaks_ind=peaks_ind, rel_height=rel_height)

        # Check if number of peaks found matches number of left bases found
        num_peaks: int = len(peaks_ind)
        num_left_bases: int = len(left_bases_ind)
        if num_peaks != num_left_bases:
            raise ValueError(
                f'''
                Number of peaks found ({num_peaks}) does not match numberof left bases found ({num_left_bases}).
                Try changing prominence and rel_height arguements.
                '''
            )

        duration: float = exp_info['duration']
        beat_freq: float = calc_beat_freq(duration, num_peaks)

        x_upbeat_lst, y_upbeat_lst = find_upbeats(
            y_data=norm_data['Mean(LL)'],
            left_bases_ind=left_bases_ind,
            peaks_ind=peaks_ind,
        )

        x_t_half_np, y_t_half_np = calc_t_half(
            x_upbeat_lst=x_upbeat_lst,
            y_upbeat_lst=y_upbeat_lst,
            kind=kind,
        )

        x_t_half_dict[roi_name] = x_t_half_np
        y_t_half_dict[roi_name] = y_t_half_np


    # calc_direction()

    # calc_phase_diff()

    # to_csv()
