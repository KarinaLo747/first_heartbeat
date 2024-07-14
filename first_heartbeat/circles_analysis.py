import pandas
import numpy
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from first_heartbeat.load_data import load_circles
from first_heartbeat.utils import create_output_dir, norm_df, Embryo, get_exp_info, real_time, calc_beat_freq
from first_heartbeat.constants import circle_roi_cols, decimal_places, manual_peak_find_csv
from first_heartbeat.plotter import time_vs_fluoresence, t_half_validation


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


def calc_direction(x_t_half_dict):

    RM_t_halfs = x_t_half_dict['mean_RM']
    RL_t_halfs = x_t_half_dict['mean_RL']
    LM_t_halfs = x_t_half_dict['mean_LM']
    LL_t_halfs = x_t_half_dict['mean_LL']

    R_delta_t_halfs = RM_t_halfs - RL_t_halfs
    # print(f'{R_delta_t_halfs = }\n')
    R_mean = round(R_delta_t_halfs.mean(), decimal_places)
    R_std = round(R_delta_t_halfs.std(), decimal_places)
    # print(f'{R_mean = }')
    # print(f'{R_std = }')

    # print()
    if R_mean > 0:
        R_res = 'lateral -> medial'
    else:
        R_res = 'medial -> lateral'
    # print(R_res)

    L_delta_t_halfs = LM_t_halfs - LL_t_halfs
    # print(f'{L_delta_t_halfs = }\n')
    L_mean = round(L_delta_t_halfs.mean(), decimal_places)
    L_std = round(L_delta_t_halfs.std(), decimal_places)
    # print(f'{L_mean = }')
    # print(f'{L_std = }')

    # print()
    if L_mean > 0:
        L_res = 'lateral -> medial'
    else:
        L_res = 'medial -> lateral'
    # print(L_res)

    print(f'Left: {L_res} ---> right: {R_res}')
    print(f'Left: {L_mean} +/- {L_std} ---> right: {R_mean} +/- {R_std}')

    return L_res, L_mean, L_std, R_res, R_mean, R_std


def peak_to_peak(x_t_half_I_sec_np: numpy.ndarray):

    I_p2p_diff = []

    peak_1 = x_t_half_I_sec_np[0]
    for i in range(1, len(x_t_half_I_sec_np)):
        peak_2 = x_t_half_I_sec_np[i]
        diff = peak_2 - peak_1
        I_p2p_diff.append(diff)
        peak_1 = peak_2

    # print(f'{len(I_p2p_diff) = }')
    # print(f'{I_p2p_diff = }')
    # print(f'{1 / np.array(I_p2p_diff) = }')

    Hz_I = 1 / np.array(I_p2p_diff)
    Hz_len = len(Hz_I)
    mean = Hz_I.mean()
    std = Hz_I.std()

    # print(f'{Hz_I = }')
    # print(f'{Hz_I.mean() = }')
    # print(f'{Hz_I.std() = }')

    return mean, std, Hz_len


def manual_peak_pick(
    data_dir: str,
    filter_regex: str = None,
    kind: str = 'linear',
    ) -> None:

    # Define name of dir for all outputs
    interim_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='interim')
    pprocessed_dir: str = create_output_dir(data_dir=data_dir, old_subdir='raw', new_subdir='processed')

    csv_stem, data = load_circles(csv_dir=data_dir, filter_regex=filter_regex)
    exp_info: Embryo = get_exp_info(csv_stem=csv_stem)
    sec_per_frame: float = exp_info.sec_per_frame

    norm_data: pandas.DataFrame = norm_df(data)

    time_vs_fluoresence(
        data=norm_data,
        sec_per_frame=sec_per_frame,
        output_dir=pprocessed_dir,
        title='All ROI',
        ext='svg',
    )

    sides: dict[str: list['str']] = {
        'Left side L and M': [circle_roi_cols[col] for col in ['mean_LL', 'mean_LM']],
        'Right side L and M': [circle_roi_cols[col] for col in ['mean_RL', 'mean_RM']],
        'Left side I': [circle_roi_cols[col] for col in ['mean_LI']],
        'Right side I': [circle_roi_cols[col] for col in ['mean_RI']],
    }

    for title, side in sides.items():
        time_vs_fluoresence(
            data=norm_data[side],
            sec_per_frame=sec_per_frame,
            output_dir=pprocessed_dir,
            title=title,
            ext='svg',
        )
        time_vs_fluoresence(
            data=norm_data[side],
            sec_per_frame=sec_per_frame,
            output_dir=pprocessed_dir,
            title=title,
            xlim=[0, 10],
            ext='svg',
        )

    x_t_half_dict: dict[str, numpy.ndarray] = {}
    y_t_half_dict: dict[str, numpy.ndarray] = {}

    roi_lst: list[str] = [
        'mean_LL',
        'mean_LI',
        'mean_LM',
        'mean_RL',
        'mean_RI',
        'mean_RM',
    ]

    roi_peak_params = {}

    for roi in roi_lst:

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
                output_dir=pprocessed_dir,
                )

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
                output_dir=pprocessed_dir,
                )

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
                x_t_half_dict[roi] = x_t_half_np
                y_t_half_dict[roi] = y_t_half_np
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

    print()
    print(f'Results for {csv_stem}')

    L_res, L_mean, L_std = np.nan, np.nan, np.nan
    R_res, R_mean, R_std = np.nan, np.nan, np.nan

    try:
        L_res, L_mean, L_std, R_res, R_mean, R_std = calc_direction(x_t_half_dict)
    except ValueError:
        print()
        print(f'---> Manually select peaks for {csv_stem}')
        with open(manual_peak_find_csv, 'a') as file:
            file.write(data_dir + '\n')
        pass

    print()

    LI = norm_data[circle_roi_cols['mean_LI']]
    RI = norm_data[circle_roi_cols['mean_RI']]

    x_np = LI.index.to_numpy()
    y_LI = LI.values
    y_RI = RI.values
    # peaks_LI = find_peak_ind(y_LI)
    # peaks_RI = find_peak_ind(y_RI)

    x_t_half_LI = x_t_half_dict['mean_LI']
    x_t_half_RI = x_t_half_dict['mean_LI']

    # peaks_LI_sec_np = real_time(x_np[peaks_LI], sec_per_frame)
    # peaks_RI_sec_np = real_time(x_np[peaks_RI], sec_per_frame)
    x_t_half_LI_sec_np = real_time(x_t_half_LI, sec_per_frame)
    x_t_half_RI_sec_np = real_time(x_t_half_RI, sec_per_frame)

    # L_Hz_mean, L_Hz_std, L_Hz_len = peak_to_peak(peaks_LI_sec_?np)
    # R_Hz_mean, R_Hz_std, R_Hz_len = peak_to_peak(peaks_RI_sec_np)
    L_Hz_mean, L_Hz_std, L_Hz_len = peak_to_peak(x_t_half_LI_sec_np)
    R_Hz_mean, R_Hz_std, R_Hz_len = peak_to_peak(x_t_half_RI_sec_np)

    Hz_diff = R_Hz_mean - L_Hz_mean
    len_Hz_diff = R_Hz_len - L_Hz_len


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
        'thalf_LM_mean': x_t_half_dict['mean_LM'].mean(),
        'thalf_LM_std': x_t_half_dict['mean_LM'].std(),
        'thalf_LL_mean': x_t_half_dict['mean_LL'].mean(),
        'thalf_LL_std': x_t_half_dict['mean_LL'].std(),
        'thalf_RM_mean': x_t_half_dict['mean_RM'].mean(),
        'thalf_RM_std': x_t_half_dict['mean_RM'].std(),
        'thalf_RL_mean': x_t_half_dict['mean_RL'].mean(),
        'thalf_RL_std': x_t_half_dict['mean_RL'].std(),
        'thalf_diff_L_mean': L_mean,
        'thalf_diff_L_std': L_std,
        'direction_L': L_res,
        'thalf_diff_R_mean': R_mean,
        'thalf_diff_R_std': R_std,
        'direction_R': R_res,
        'Hz_L_mean': L_Hz_mean,
        'Hz_L_std': L_Hz_std,
        'Hz_L_len': L_Hz_len,
        'Hz_R_mean': R_Hz_mean,
        'Hz_R_std': R_Hz_std,
        'Hz_R_len': R_Hz_len,
        'Hz_diff': Hz_diff,
        'len_Hz_diff': len_Hz_diff,
    }

    return results_dict

        ## JSON structure
        # result = {
        #     roi: {
        #         prominence,
        #         rel_height,
        #     }
        # }


# def run_analysis(
#     data_dir: str,
#     filter_regex: str = None,
#     prominence: float = 0.5,
#     rel_height: float = 0.90,
#     kind: str = 'linear',
#     ) -> None:

#     # Define name of dir for all outputs
#     output_dir: str = create_output_dir(data_dir=data_dir)

#     csv_stem, data = load_circles(csv_dir=data_dir, filter_regex=filter_regex)
#     exp_info: Embryo = get_exp_info(csv_stem=csv_stem)
#     sec_per_frame: float = exp_info.sec_per_frame

#     norm_data: pandas.DataFrame = norm_df(data)

#     time_vs_fluoresence(
#         data=norm_data,
#         sec_per_frame=sec_per_frame,
#         output_dir=output_dir,
#         title='All ROI',
#         ext='png',
#     )

#     sides: dict[str: list['str']] = {
#         'Left side L and M': [circle_roi_cols[col] for col in ['mean_LL', 'mean_LM']],
#         'Right side L and M': [circle_roi_cols[col] for col in ['mean_RL', 'mean_RM']],
#         'Left side I': [circle_roi_cols[col] for col in ['mean_LI']],
#         'Right side I': [circle_roi_cols[col] for col in ['mean_RI']],
#     }

#     for title, side in sides.items():
#         time_vs_fluoresence(
#             data=norm_data[side],
#             sec_per_frame=sec_per_frame,
#             output_dir=output_dir,
#             title=title,
#             ext='png',
#         )
#         time_vs_fluoresence(
#             data=norm_data[side],
#             sec_per_frame=sec_per_frame,
#             output_dir=output_dir,
#             title=title,
#             xlim=[0, 10],
#             ext='png',
#         )

#     peaks_dict: dict[str, numpy.ndarray] = {}
#     left_bases_dict: dict[str, numpy.ndarray] = {}
#     beat_freq_dict: dict[str, float] = {}

#     beat_freq_roi: list[str] = [
#         'mean_LL',
#         'mean_LI',
#         'mean_LM',
#         'mean_RL',
#         'mean_RI',
#         'mean_RM',
#     ]

#     for roi_name in beat_freq_roi:

#         y_data: pandas.Series = norm_data[circle_roi_cols[roi_name]]
#         peaks_ind: numpy.ndarray = find_peak_ind(y_data=y_data, prominence=prominence)
#         left_bases_ind: numpy.ndarray = find_peak_base_ind(y_data=y_data, peaks_ind=peaks_ind, rel_height=rel_height)

#         peaks_dict[roi_name] = peaks_ind
#         left_bases_dict[roi_name] = left_bases_ind

#         # Check if number of peaks found matches number of left bases found
#         num_peaks: int = len(peaks_ind)

#         # num_left_bases: int = len(left_bases_ind)
#         # if num_peaks != num_left_bases:
#         #     raise ValueError(
#         #         f'''
#         #         Number of peaks found ({num_peaks}) does not match numberof left bases found ({num_left_bases}).
#         #         Try changing prominence and rel_height arguements.
#         #         '''
#         #     )

#         duration: float = exp_info.duration
#         beat_freq: float = calc_beat_freq(duration, num_peaks)
#         beat_freq_dict[roi_name] = beat_freq

#     x_t_half_dict: dict[str, numpy.ndarray] = {}
#     y_t_half_dict: dict[str, numpy.ndarray] = {}

#     t_half_roi = [
#         'mean_LL',
#         'mean_LM',
#         'mean_RL',
#         'mean_RM',
#     ]

#     for roi_name in t_half_roi:

#         x_np: numpy.ndarray = norm_data[circle_roi_cols[roi_name]].index.to_numpy()
#         y_np: numpy.ndarray = norm_data[circle_roi_cols[roi_name]].values

#         # Setup figure to visualise result
#         fig, ax = plt.subplots(
#             2, 2,
#             sharex=True,
#             sharey=True,
#         )

#         # Formatting for linewidth (lw), size (s) and markersize (s)
#         lw, s, ms = 1, 15, 5

#         # Plot original time vs normalised fluoresence intensity
#         ax[0, 0].set_title('Original plot')
#         for i, j in zip([0, 0, 1], [0, 1, 1]):
#             ax[i, j].plot(real_time(x_np, sec_per_frame), y_np, c='k', lw=lw, label='Original plot')

#         peaks_ind = peaks_dict[roi_name]
#         left_bases_ind = left_bases_dict[roi_name]
#         sec_per_frame = exp_info.sec_per_frame

#         # Plot left bases and peaks
#         ax[0, 1].set_title('Left bases and peaks')
#         ax[0, 1].scatter(real_time(x_np[peaks_ind], sec_per_frame), y_np[[peaks_ind]], c='darkblue', marker='*', s=s)
#         ax[0, 1].scatter(real_time(x_np[left_bases_ind], sec_per_frame), y_np[left_bases_ind], c='darkblue', marker='^', s=s)


#         x_upbeat_lst, y_upbeat_lst = find_upbeats(
#             y_data=norm_data[circle_roi_cols[roi_name]],
#             left_bases_ind=left_bases_dict[roi_name],
#             peaks_ind=peaks_dict[roi_name],
#         )

#         x_t_half_np, y_t_half_np = calc_t_half(
#             x_upbeat_lst=x_upbeat_lst,
#             y_upbeat_lst=y_upbeat_lst,
#             kind=kind,
#         )

#         x_t_half_dict[roi_name] = x_t_half_np
#         y_t_half_dict[roi_name] = y_t_half_np

#         # Plot ubbeats only showing data points
#         ax[1, 0].set_title('Upbeats')
#         for x_wind, y_wind in zip(x_upbeat_lst, y_upbeat_lst):
#             ax[1, 0].plot(real_time(x_wind, sec_per_frame), y_wind, lw=lw, color='k', marker='.', ms=ms)

#         # Plot position of t_half
#         ax[1, 1].set_title('$t_{1/2}$')
#         ax[1, 1].scatter(real_time(x_t_half_np, sec_per_frame), y_t_half_np, c='red', marker='x', s=s)

#         # Figure formatting
#         cut = exp_info.cut
#         fig.suptitle(f'{cut.title()} {title}')
#         fig.supxlabel('Time / s')
#         fig.supylabel('Normalised fluoresence intensity / a.u.')
#         plt.tight_layout()
#         plt.savefig(output_dir + f'peak_checker-{roi_name}.png')
#         plt.close()

#     print()
#     print(f'Results for {csv_stem}')

#     L_res, L_mean, L_std = np.nan, np.nan, np.nan
#     R_res, R_mean, R_std = np.nan, np.nan, np.nan

#     try:
#         L_res, L_mean, L_std, R_res, R_mean, R_std = calc_direction(x_t_half_dict)
#     except ValueError:
#         print()
#         print(f'---> Manually select peaks for {csv_stem}')
#         with open(manual_peak_find_csv, 'a') as file:
#             file.write(data_dir + '\n')
#         pass

#     print()

#     LI = norm_data[circle_roi_cols['mean_LI']]
#     RI = norm_data[circle_roi_cols['mean_RI']]

#     x_np = LI.index.to_numpy()
#     y_LI = LI.values
#     y_RI = RI.values
#     peaks_LI = find_peak_ind(y_LI)
#     peaks_RI = find_peak_ind(y_RI)

#     peaks_LI_sec_np = real_time(x_np[peaks_LI], sec_per_frame)
#     peaks_RI_sec_np = real_time(x_np[peaks_RI], sec_per_frame)

#     L_Hz_mean, L_Hz_std, L_Hz_len = peak_to_peak(peaks_LI_sec_np)
#     R_Hz_mean, R_Hz_std, R_Hz_len = peak_to_peak(peaks_RI_sec_np)

#     Hz_diff = R_Hz_mean - L_Hz_mean
#     len_Hz_diff = R_Hz_len - L_Hz_len


#     results_dict: dict[str, any] = {
#         'date': exp_info.date,
#         'mouse_line': exp_info.mouse_line,
#         'dpc': exp_info.dpc,
#         'exp': exp_info.exp,
#         'embryo': exp_info.embryo,
#         'mag': exp_info.mag,
#         'total_frames': exp_info.total_frames,
#         'cut': exp_info.cut,
#         'section': exp_info.section,
#         'repeat': exp_info.repeat,
#         'linestep': exp_info.linestep,
#         'stage': exp_info.stage,
#         'duration': exp_info.duration,
#         'sec_per_frame': exp_info.sec_per_frame,
#         'thalf_LM_mean': x_t_half_dict['mean_LM'].mean(),
#         'thalf_LM_std': x_t_half_dict['mean_LM'].std(),
#         'thalf_LL_mean': x_t_half_dict['mean_LL'].mean(),
#         'thalf_LL_std': x_t_half_dict['mean_LL'].std(),
#         'thalf_RM_mean': x_t_half_dict['mean_RM'].mean(),
#         'thalf_RM_std': x_t_half_dict['mean_RM'].std(),
#         'thalf_RL_mean': x_t_half_dict['mean_RL'].mean(),
#         'thalf_RL_std': x_t_half_dict['mean_RL'].std(),
#         'thalf_diff_L_mean': L_mean,
#         'thalf_diff_L_std': L_std,
#         'direction_L': L_res,
#         'thalf_diff_R_mean': R_mean,
#         'thalf_diff_R_std': R_std,
#         'direction_R': R_res,
#         'Hz_L_mean': L_Hz_mean,
#         'Hz_L_std': L_Hz_std,
#         'Hz_L_len': L_Hz_len,
#         'Hz_R_mean': R_Hz_mean,
#         'Hz_R_std': R_Hz_std,
#         'Hz_R_len': R_Hz_len,
#         'Hz_diff': Hz_diff,
#         'len_Hz_diff': len_Hz_diff,
#     }

    # return results_dict


    # calc_phase_diff()

    # to_csv()
