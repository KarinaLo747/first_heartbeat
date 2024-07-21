import pandas
import numpy
import numpy as np
import json
import matplotlib.pyplot as plt
from functools import partial
from dataclasses import dataclass
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from first_heartbeat.load_data import load_circles
from first_heartbeat.utils import create_output_dir, norm_df, Embryo, get_exp_info, real_time, calc_beat_freq
from first_heartbeat.constants import circle_roi_cols, decimal_places, manual_peak_find_csv
from first_heartbeat.plotter import time_vs_fluoresence, t_half_validation


def find_peak_ind(y_data: any, prominence: float = 0.5, height: float = None, distance: float = 4.0) -> numpy.ndarray:
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
        distance=distance,
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


@dataclass
class THalf:
    thalf_LL: numpy.ndarray
    thalf_LI: numpy.ndarray
    thalf_LM: numpy.ndarray
    thalf_RL: numpy.ndarray
    thalf_RI: numpy.ndarray
    thalf_RM: numpy.ndarray

    def calc_direction(self):
        """
        outcome_dict = {
            1: 'Left to Right',
            2: 'Right to Left',
            3: 'Lateral to Medial',
            4: 'Medial to Lateral',
            5: 'Error calculating',
        }
        """

        # Right side
        if self.thalf_RM.shape == self.thalf_RL.shape:
            self.Dthalf_R = self.thalf_RM - self.thalf_RL
            self.Dthalf_R_method = 'RM - RL'

        elif self.thalf_RM.shape == self.thalf_RI.shape:
            self.Dthalf_R = self.thalf_RM - self.thalf_RI
            self.Dthalf_R_method = 'RM - RI'

        elif self.thalf_RI.shape == self.thalf_RL.shape:
            self.Dthalf_R = self.thalf_RI - self.thalf_RL
            self.Dthalf_R_method = 'RI - RL'

        else:
            self.Dthalf_R = np.nan
            self.Dthalf_R_method = None

        self.Dthalf_R_mean = self.Dthalf_R.mean()
        self.Dthalf_R_std = self.Dthalf_R.std()

        # Left side
        if self.thalf_LM.shape == self.thalf_LL.shape:
            self.Dthalf_L = self.thalf_LM - self.thalf_LL
            self.Dthalf_L_method = 'LM - LL'

        elif self.thalf_LM.shape == self.thalf_LI.shape:
            self.Dthalf_L = self.thalf_LM - self.thalf_LI
            self.Dthalf_L_method = 'LM - LI'

        elif self.thalf_LI.shape == self.thalf_LL.shape:
            self.Dthalf_L = self.thalf_LI - self.thalf_LL
            self.Dthalf_L_method = 'LI - LL'

        else:
            self.Dthalf_L = np.nan
            self.Dthalf_L_method = None

        self.Dthalf_L_mean = self.Dthalf_L.mean()
        self.Dthalf_L_std = self.Dthalf_L.std()

        # 1: 'Left to Right'
        if self.Dthalf_R_mean < 0 and self.Dthalf_L_mean > 0:
            self.direction = 1

        # 2: 'Right to Left'
        elif self.Dthalf_R_mean > 0 and self.Dthalf_L_mean < 0:
            self.direction = 2

        # 3: 'Lateral to Medial'
        elif self.Dthalf_R_mean > 0 and self.Dthalf_L_mean > 0:
            self.direction = 3

        # 4: 'Medial to Lateral'
        elif self.Dthalf_R_mean < 0 and self.Dthalf_L_mean < 0:
            self.direction = 4

        # 5: 'Error calculating'
        else:
            self.direction = 5


@dataclass
class Rhythmicity:
    thalf: numpy.ndarray

    def calc_rhythmicity(self):
        peak_diff = np.diff(self.thalf)
        self.Hz = 1 / peak_diff
        self.Hz_len = len(self.Hz)
        self.Hz_mean = self.Hz.mean()
        self.Hz_std = self.Hz.std()
        self.Hz_min = self.Hz.min()
        self.Hz_max = self.Hz.max()
        self.Hz_range = np.ptp(self.Hz)


def calc_direction(x_t_half_dict):

    # th = t_half
    RM_t_halfs: numpy.ndarray = x_t_half_dict['mean_RM']
    RI_t_halfs: numpy.ndarray = x_t_half_dict['mean_RI']
    RL_t_halfs: numpy.ndarray = x_t_half_dict['mean_RL']
    LM_t_halfs: numpy.ndarray = x_t_half_dict['mean_LM']
    LI_t_halfs: numpy.ndarray = x_t_half_dict['mean_LI']
    LL_t_halfs: numpy.ndarray = x_t_half_dict['mean_LL']

    try:
        # Dth = Delta t_half
        R_Dth = RM_t_halfs - RL_t_halfs
        thalf_diff_R_method = 'RM - RL'
    except ValueError:
        try:
            print('Trying RM - RI')
            thalf_diff_R_method = 'RM - RI'
            R_delta_t_halfs = RM_t_halfs - RI_t_halfs
        except ValueError:
            print('Trying RI - RL')
            thalf_diff_R_method = 'RI - RL'
            R_delta_t_halfs = RI_t_halfs - RL_t_halfs
    except:
        thalf_diff_R_method = None
        R_delta_t_halfs = np.nan
    R_mean = round(R_delta_t_halfs.mean(), decimal_places)
    R_std = round(R_delta_t_halfs.std(), decimal_places)

    if R_mean > 0:
        R_res = 'lateral -> medial'
    else:
        R_res = 'medial -> lateral'

    try:
        L_delta_t_halfs = LM_t_halfs - LL_t_halfs
        thalf_diff_L_method = 'LM - LL'
    except ValueError:
        try:
            print('Trying LM - LI')
            thalf_diff_L_method = 'LM - LI'
            L_delta_t_halfs = LM_t_halfs - LI_t_halfs
        except ValueError:
            print('Trying LI - LL')
            thalf_diff_L_method = 'LI - LL'
            L_delta_t_halfs = LI_t_halfs - LL_t_halfs
    except:
        thalf_diff_L_method = None
        L_delta_t_halfs = np.nan
    L_mean = round(L_delta_t_halfs.mean(), decimal_places)
    L_std = round(L_delta_t_halfs.std(), decimal_places)

    # print()
    if L_mean > 0:
        L_res = 'lateral -> medial'
    else:
        L_res = 'medial -> lateral'

    overall_direction = None

    # outcome_dict = {
    #     1: 'Left to Right',
    #     2: 'Right to Left',
    #     3: 'Lateral to Medial',
    #     4: 'Medial to Lateral',
    # }

    # 1: 'Left to Right',
    if R_mean < 0 and L_mean > 0:
        overall_direction = 1

    # 2: 'Right to Left',
    if R_mean > 0 and L_mean < 0:
        overall_direction = 2

    # 3: 'Lateral to Medial',
    if R_mean > 0 and L_mean > 0:
        overall_direction = 3

    # 4: 'Medial to Lateral',
    if R_mean < 0 and L_mean < 0:
        overall_direction = 4

    print(f'Left: {L_res} ---> right: {R_res}')
    print(f'Left: {L_mean} +/- {L_std} ---> right: {R_mean} +/- {R_std}')
    print(f'{overall_direction = }')

    return L_res, L_mean, L_std, R_res, R_mean, R_std, thalf_diff_L_method, thalf_diff_R_method, overall_direction


def peak_to_peak(x_t_half_I_sec_np: numpy.ndarray):

    I_p2p_diff = []

    peak_1 = x_t_half_I_sec_np[0]
    for i in range(1, len(x_t_half_I_sec_np)):
        peak_2 = x_t_half_I_sec_np[i]
        diff = peak_2 - peak_1
        I_p2p_diff.append(diff)
        peak_1 = peak_2

    Hz_I = 1 / np.array(I_p2p_diff)
    Hz_len = len(Hz_I)
    mean = Hz_I.mean()
    std = Hz_I.std()

    return mean, std, Hz_len


def manual_peak_pick(
    data_dir: str,
    embryo: int,
    filter_regex: str = None,
    kind: str = 'linear',
    ) -> None:

    interim_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='interim')
    processed_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='processed')

    csv_stem, data = load_circles(csv_dir=data_dir, filter_regex=filter_regex)
    exp_info: Embryo = get_exp_info(csv_stem=csv_stem)
    sec_per_frame: float = exp_info.sec_per_frame

    to_sec = partial(real_time, sec_per_frame=sec_per_frame)

    norm_data: pandas.DataFrame = norm_df(data)

    save_svg = partial(
        time_vs_fluoresence,
        sec_per_frame=sec_per_frame,
        output_dir=processed_dir,
        ext='svg',
    )

    save_svg(data=norm_data, title='All ROI')

    sides: dict[str: list['str']] = {
        'Left side L and M': [circle_roi_cols[col] for col in ['mean_LL', 'mean_LM']],
        'Right side L and M': [circle_roi_cols[col] for col in ['mean_RL', 'mean_RM']],
        'Left side I': [circle_roi_cols['mean_LI']],
        'Right side I': [circle_roi_cols['mean_RI']],
    }

    for title, side in sides.items():
        save_svg(data=norm_data[side], title=title)
        save_svg(data=norm_data[side], title=title, xlim=[0, 10])

    x_t_half_dict: dict[str, numpy.ndarray] = {}
    y_t_half_dict: dict[str, numpy.ndarray] = {}

    roi_dict: dict[str, str] = {
        'mean_LL': 'thalf_LL',
        'mean_LI': 'thalf_LI',
        'mean_LM': 'thalf_LM',
        'mean_RL': 'thalf_RL',
        'mean_RI': 'thalf_RI',
        'mean_RM': 'thalf_RM',
    }

    roi_peak_params = {}

    for roi, thalf_name in roi_dict.items():

        # Start state
        prominence: float = 0.5
        rel_height: float = 0.90
        abort = False

        # Load data
        col_name: str = circle_roi_cols[roi]
        y_data: pandas.Series = norm_data[col_name]
        x_np = y_data.index.to_numpy()
        y_np = y_data.values

        while True:

            # Find peaks and left bases
            peaks_ind: numpy.ndarray = find_peak_ind(y_data=y_data, prominence=prominence)
            left_bases_ind: numpy.ndarray = find_peak_base_ind(y_data=y_data, peaks_ind=peaks_ind, rel_height=rel_height)

            # Calculate upbeat
            x_upbeat_coord_lst, y_upbeat_coord_lst = find_upbeats(
                y_data=norm_data[col_name],
                left_bases_ind=left_bases_ind,
                peaks_ind=peaks_ind,
            )

            # Calculate t_half
            x_t_half_np, y_t_half_np = calc_t_half(
                x_upbeat_lst=x_upbeat_coord_lst,
                y_upbeat_lst=y_upbeat_coord_lst,
                kind=kind,
            )

            t_half_validation(
                roi=roi,
                prominence=prominence,
                rel_height=rel_height,
                x_np=x_np,
                y_np=y_np,
                peaks_ind=peaks_ind,
                left_bases_ind=left_bases_ind,
                x_upbeat_coord_lst=x_upbeat_coord_lst,
                y_upbeat_coord_lst=y_upbeat_coord_lst,
                x_t_half_np=x_t_half_np,
                y_t_half_np=y_t_half_np,
                sec_per_frame=sec_per_frame,
                output_dir=processed_dir,
                )

            print(f'Num. peaks = {len(peaks_ind)}')
            prominence_answer: str = input(f'{roi}: peaks satisfactory? [y/n/a]')

            if prominence_answer.lower() in ('n', 'no'):

                print(f'{roi}: Let us try another prominence...')

                try:
                    prominence = float(input(f'{roi}: Enter prominence (float between 0.0 and 1.0):'))

                except ValueError:
                    print(f'---> Please enter a float <---')
                    continue

            elif prominence_answer.lower() in ('y', 'yes', 'ye', 'es', 'ys'):
                print(f'{roi}: Success! Final {prominence = }')
                break

            elif prominence_answer.lower() in ('abort', 'ab', 'a'):
                print(f'{roi}: Aborting')
                abort = True
                break

            else:
                print(f'---> Please enter y or n <---')
                continue


        while True and abort != True:

            # Find peaks and left bases
            peaks_ind: numpy.ndarray = find_peak_ind(y_data=y_data, prominence=prominence)
            left_bases_ind: numpy.ndarray = find_peak_base_ind(y_data=y_data, peaks_ind=peaks_ind, rel_height=rel_height)

            # Calculate upbeat
            x_upbeat_coord_lst, y_upbeat_coord_lst = find_upbeats(
                y_data=norm_data[col_name],
                left_bases_ind=left_bases_ind,
                peaks_ind=peaks_ind,
            )

            # Calculate t_half
            x_t_half_np, y_t_half_np = calc_t_half(
                x_upbeat_lst=x_upbeat_coord_lst,
                y_upbeat_lst=y_upbeat_coord_lst,
                kind=kind,
            )

            t_half_validation(
                roi=roi,
                prominence=prominence,
                rel_height=rel_height,
                x_np=x_np,
                y_np=y_np,
                peaks_ind=peaks_ind,
                left_bases_ind=left_bases_ind,
                x_upbeat_coord_lst=x_upbeat_coord_lst,
                y_upbeat_coord_lst=y_upbeat_coord_lst,
                x_t_half_np=x_t_half_np,
                y_t_half_np=y_t_half_np,
                sec_per_frame=sec_per_frame,
                output_dir=processed_dir,
                )

            print(f'Num. peaks = {len(peaks_ind)}')
            rel_height_answer: str = input(f'{roi}: t_half satisfactory? [y/n/a]')

            if rel_height_answer.lower() in ('n', 'no'):

                print(f'{roi}: Let us try another rel_height...')

                try:
                    rel_height = float(input(f'{roi}: Enter rel_height (float):'))

                except ValueError:
                    print(f'---> Please enter a float <---')
                    continue

            elif rel_height_answer.lower() in ('y', 'yes', 'ye', 'es', 'ys'):
                print(f'{roi}: Success! Final {rel_height = }')
                x_t_half_dict[thalf_name] = x_t_half_np
                y_t_half_dict[thalf_name] = y_t_half_np
                break

            elif rel_height_answer.lower() in ('abort', 'ab', 'a'):
                print(f'{roi}: Aborting')
                break

            else:
                print(f'---> Please enter y or n <---')
                continue

        roi_peak_params[roi] = {
            'prominence': prominence,
            'rel_height': rel_height,
        }

    json_loc = interim_dir + 'peak_params.json'
    with open(json_loc, 'w') as f:
        json.dump(roi_peak_params, f, indent=4)

    thalf: THalf = THalf(**x_t_half_dict)
    thalf.calc_direction()

    rhyth_L = Rhythmicity(thalf=to_sec(thalf.thalf_LI))
    rhyth_L.calc_rhythmicity()
    rhyth_R = Rhythmicity(thalf=to_sec(thalf.thalf_RI))
    rhyth_R.calc_rhythmicity()

    results_dict: dict[str, any] = {
        'date': exp_info.date,
        'mouse_line': exp_info.mouse_line,
        'dpc': exp_info.dpc,
        'exp': exp_info.exp,
        'embryo': exp_info.embryo,
        'mag': exp_info.mag,
        'total_frames': exp_info.total_frames,
        'cut': exp_info.cut,
        'section': exp_info.section,
        'repeat': exp_info.repeat,
        'linestep': exp_info.linestep,
        'stage': exp_info.stage,
        'duration': exp_info.duration,
        'sec_per_frame': exp_info.sec_per_frame,
        'thalf_LL': thalf.thalf_LL.tolist(),
        'thalf_LI': thalf.thalf_LI.tolist(),
        'thalf_LM': thalf.thalf_LM.tolist(),
        'thalf_RL': thalf.thalf_RL.tolist(),
        'thalf_RI': thalf.thalf_RI.tolist(),
        'thalf_RM': thalf.thalf_RM.tolist(),
        'thalf_LM_mean': thalf.thalf_LM.mean(),
        'thalf_LM_std': thalf.thalf_LM.std(),
        'thalf_LL_mean': thalf.thalf_LL.mean(),
        'thalf_LL_std': thalf.thalf_LL.std(),
        'thalf_RM_mean': thalf.thalf_RM.mean(),
        'thalf_RM_std': thalf.thalf_RM.std(),
        'thalf_RL_mean': thalf.thalf_RL.mean(),
        'thalf_RL_std': thalf.thalf_RL.std(),
        'Dthalf_L_mean': thalf.Dthalf_L.mean(),
        'Dthalf_L_std': thalf.Dthalf_L.std(),
        'Dthalf_L_method': thalf.Dthalf_L_method,
        'Dthalf_R_mean': thalf.Dthalf_R.mean(),
        'Dthalf_R_std': thalf.Dthalf_R.std(),
        'Dthalf_R_method': thalf.Dthalf_R_method,
        'direction': thalf.direction,
        'Hz_L' : rhyth_L.Hz.tolist(),
        'Hz_L_len' : rhyth_L.Hz_len,
        'Hz_L_mean' : rhyth_L.Hz_mean,
        'Hz_L_std' : rhyth_L.Hz_std,
        'Hz_L_min' : rhyth_L.Hz_min,
        'Hz_L_max' : rhyth_L.Hz_max,
        'Hz_L_range' : rhyth_L.Hz_range,
        'Hz_R' : rhyth_R.Hz.tolist(),
        'Hz_R_len' : rhyth_R.Hz_len,
        'Hz_R_mean' : rhyth_R.Hz_mean,
        'Hz_R_std' : rhyth_R.Hz_std,
        'Hz_R_min' : rhyth_R.Hz_min,
        'Hz_R_max' : rhyth_R.Hz_max,
        'Hz_R_range' : rhyth_R.Hz_range,
    }

    return results_dict


def load_peak_params(
    data_dir: str,
    filter_regex: str = None,
    kind: str = 'linear',
    plot_graphs: bool = False,
    ) -> None:
    """_summary_

    outcome_dict = {
        1: 'Left to Right',
        2: 'Right to Left',
        3: 'Lateral to Medial',
        3: 'Medial to Lateral',
    }

    Args:
        data_dir (str): _description_
        filter_regex (str, optional): _description_. Defaults to None.
        kind (str, optional): _description_. Defaults to 'linear'.
    """

    interim_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='interim')
    processed_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='processed')

    csv_stem, data = load_circles(csv_dir=data_dir, filter_regex=filter_regex)
    exp_info: Embryo = get_exp_info(csv_stem=csv_stem)
    sec_per_frame: float = exp_info.sec_per_frame

    to_sec = partial(real_time, sec_per_frame=sec_per_frame)

    norm_data: pandas.DataFrame = norm_df(data)

    if plot_graphs:

        save_svg = partial(
            time_vs_fluoresence,
            sec_per_frame=sec_per_frame,
            output_dir=processed_dir,
            ext='svg',
        )

        save_svg(data=norm_data, title='All ROI')

        sides: dict[str: list['str']] = {
            'Left side L and M': [circle_roi_cols[col] for col in ['mean_LL', 'mean_LM']],
            'Right side L and M': [circle_roi_cols[col] for col in ['mean_RL', 'mean_RM']],
            'Left side I': [circle_roi_cols['mean_LI']],
            'Right side I': [circle_roi_cols['mean_RI']],
        }

        for title, side in sides.items():
            save_svg(data=norm_data[side], title=title)
            save_svg(data=norm_data[side], title=title, xlim=[0, 10])

    x_t_half_dict: dict[str, numpy.ndarray] = {}
    y_t_half_dict: dict[str, numpy.ndarray] = {}

    roi_dict: dict[str, str] = {
        'mean_LL': 'thalf_LL',
        'mean_LI': 'thalf_LI',
        'mean_LM': 'thalf_LM',
        'mean_RL': 'thalf_RL',
        'mean_RI': 'thalf_RI',
        'mean_RM': 'thalf_RM',
    }

    json_path: str = interim_dir + 'peak_params.json'
    with open(json_path) as f:
        peak_params: dict[dict[str, float]] = json.load(f)

    for roi, thalf_name in roi_dict.items():

        # Loaded params
        roi_param: dict[str, float] = peak_params[roi]
        prominence: float = roi_param['prominence']
        rel_height: float = roi_param['rel_height']

        # Load data
        col_name: str = circle_roi_cols[roi]
        y_data: pandas.Series = norm_data[col_name]
        x_np = y_data.index.to_numpy()
        y_np = y_data.values

        peaks_ind: numpy.ndarray = find_peak_ind(y_data=y_data, prominence=prominence)
        left_bases_ind: numpy.ndarray = find_peak_base_ind(y_data=y_data, peaks_ind=peaks_ind, rel_height=rel_height)

        # Calculate upbeat
        x_upbeat_coord_lst, y_upbeat_coord_lst = find_upbeats(
            y_data=norm_data[col_name],
            left_bases_ind=left_bases_ind,
            peaks_ind=peaks_ind,
        )

        # Calculate t_half
        x_t_half_np, y_t_half_np = calc_t_half(
            x_upbeat_lst=x_upbeat_coord_lst,
            y_upbeat_lst=y_upbeat_coord_lst,
            kind=kind,
        )

        if plot_graphs:
            t_half_validation(
                roi=roi,
                prominence=prominence,
                rel_height=rel_height,
                x_np=x_np,
                y_np=y_np,
                peaks_ind=peaks_ind,
                left_bases_ind=left_bases_ind,
                x_upbeat_coord_lst=x_upbeat_coord_lst,
                y_upbeat_coord_lst=y_upbeat_coord_lst,
                x_t_half_np=x_t_half_np,
                y_t_half_np=y_t_half_np,
                sec_per_frame=sec_per_frame,
                output_dir=processed_dir,
                show=False,
                )

        x_t_half_dict[thalf_name] = x_t_half_np
        y_t_half_dict[thalf_name] = y_t_half_np

    thalf: THalf = THalf(**x_t_half_dict)
    thalf.calc_direction()

    rhyth_L = Rhythmicity(thalf=to_sec(thalf.thalf_LI))
    rhyth_L.calc_rhythmicity()
    rhyth_R = Rhythmicity(thalf=to_sec(thalf.thalf_RI))
    rhyth_R.calc_rhythmicity()

    results_dict: dict[str, any] = {
        'date': exp_info.date,
        'mouse_line': exp_info.mouse_line,
        'dpc': exp_info.dpc,
        'exp': exp_info.exp,
        'embryo': exp_info.embryo,
        'mag': exp_info.mag,
        'total_frames': exp_info.total_frames,
        'cut': exp_info.cut,
        'section': exp_info.section,
        'repeat': exp_info.repeat,
        'linestep': exp_info.linestep,
        'stage': exp_info.stage,
        'duration': exp_info.duration,
        'sec_per_frame': exp_info.sec_per_frame,
        'thalf_LL': thalf.thalf_LL.tolist(),
        'thalf_LI': thalf.thalf_LI.tolist(),
        'thalf_LM': thalf.thalf_LM.tolist(),
        'thalf_RL': thalf.thalf_RL.tolist(),
        'thalf_RI': thalf.thalf_RI.tolist(),
        'thalf_RM': thalf.thalf_RM.tolist(),
        'thalf_LM_mean': thalf.thalf_LM.mean(),
        'thalf_LM_std': thalf.thalf_LM.std(),
        'thalf_LL_mean': thalf.thalf_LL.mean(),
        'thalf_LL_std': thalf.thalf_LL.std(),
        'thalf_RM_mean': thalf.thalf_RM.mean(),
        'thalf_RM_std': thalf.thalf_RM.std(),
        'thalf_RL_mean': thalf.thalf_RL.mean(),
        'thalf_RL_std': thalf.thalf_RL.std(),
        'Dthalf_L_mean': thalf.Dthalf_L.mean(),
        'Dthalf_L_std': thalf.Dthalf_L.std(),
        'Dthalf_L_method': thalf.Dthalf_L_method,
        'Dthalf_R_mean': thalf.Dthalf_R.mean(),
        'Dthalf_R_std': thalf.Dthalf_R.std(),
        'Dthalf_R_method': thalf.Dthalf_R_method,
        'direction': thalf.direction,
        'Hz_L' : rhyth_L.Hz.tolist(),
        'Hz_L_len' : rhyth_L.Hz_len,
        'Hz_L_mean' : rhyth_L.Hz_mean,
        'Hz_L_std' : rhyth_L.Hz_std,
        'Hz_L_min' : rhyth_L.Hz_min,
        'Hz_L_max' : rhyth_L.Hz_max,
        'Hz_L_range' : rhyth_L.Hz_range,
        'Hz_R' : rhyth_R.Hz.tolist(),
        'Hz_R_len' : rhyth_R.Hz_len,
        'Hz_R_mean' : rhyth_R.Hz_mean,
        'Hz_R_std' : rhyth_R.Hz_std,
        'Hz_R_min' : rhyth_R.Hz_min,
        'Hz_R_max' : rhyth_R.Hz_max,
        'Hz_R_range' : rhyth_R.Hz_range,
    }

    return results_dict
