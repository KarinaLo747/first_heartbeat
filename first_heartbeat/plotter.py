import numpy
import logging
import pandas
import matplotlib.pyplot as plt
from first_heartbeat.utils import real_time


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


def time_vs_fluoresence(
    data: pandas.DataFrame,
    sec_per_frame: float,
    output_dir: str,
    title: str = None,
    xlim: list[int] = None,
    ext: str = 'svg',
) -> None:
    """Plots time vs normalised fluorescence intensity. If input DataFrame contains multiple columns,
    each column is plotted as a separate line labelled with the column header name in the legend.

    Note:
        The frame number is assumed to be stored in DataFrame index column. The frame number is converted
        to real time in this function with the inputted sec_per_frame argument.

    Args:
        data (pandas.DataFrame): DataFrame containing fluorescence intensity data as columns indexed by frame number.
        sec_per_frame (float): Calculated seconds per frame used to scale frame number by.
        output_dir (str): The location to save the figure to.
        title (str, optional): Title of the figure. Defaults to None.
        xlim (list[float], optional): Range of x-axis to show and save. Defaults to None.
        ext (str, optional): File extension of saved figure, such as \'png\' or \'svg\'. Defaults to 'svg'.
    """

    fig, ax = plt.subplots()

    for col in data:
        x: numpy.ndarray = data[col].index * sec_per_frame
        y: numpy.ndarray = data[col].values
        ax.plot(x, y, label=col)

    ax.set_title(title)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Normalised fluorescence intensity / a.u.')

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        title += f'-xlim_{xlim[0]}_to_{xlim[1]}'

    plt.legend(loc='lower right')
    plt.tight_layout()
    save_loc = output_dir + title.lower().replace(' ', '_') + '-time_vs_fluorescence_intensity' + '.' + ext.lower()
    plt.savefig(save_loc)
    plt.close()
    logger.info(f'Saved {save_loc}')


def t_half_validation(
    roi: str,
    prominence: float,
    rel_height: float,
    x_np: numpy.ndarray,
    y_np: numpy.ndarray,
    peaks_ind: numpy.ndarray,
    left_bases_ind: numpy.ndarray,
    x_upbeat_coord_lst: list[numpy.ndarray],
    y_upbeat_coord_lst: list[numpy.ndarray],
    x_t_half_np: numpy.ndarray,
    y_t_half_np: numpy.ndarray,
    sec_per_frame: float,
    output_dir: str,
    ext: str = 'svg',
    show: bool = True,
    ) -> None:

        # Setup figure
        fig, ax = plt.subplots(
            2, 2,
            sharex=True,
            sharey=True,
        )

        # Formatting for linewidth (lw), size (s) and markersize (s)
        lw, s, ms = 1, 15, 5

        # Plot original time vs normalised fluoresence intensity
        ax[0, 0].set_title('Original plot')
        for i, j in zip([0, 0, 1], [0, 1, 1]):
            ax[i, j].plot(real_time(x_np, sec_per_frame), y_np, c='k', lw=lw, label='Original plot')

        # Plot left bases and peaks
        ax[0, 1].set_title('Left bases and peaks')
        ax[0, 1].scatter(real_time(x_np[peaks_ind], sec_per_frame), y_np[peaks_ind], c='green', marker='*', s=s)
        ax[0, 1].scatter(real_time(x_np[left_bases_ind], sec_per_frame), y_np[left_bases_ind], c='blue', marker='^', s=s)

        # Plot ubbeats only showing data points
        ax[1, 0].set_title('Upbeats')
        for x_coord, y_coord in zip(x_upbeat_coord_lst, y_upbeat_coord_lst):
            ax[1, 0].plot(real_time(x_coord, sec_per_frame), y_coord, lw=lw, color='k', marker='.', ms=ms)

        # Plot position of t_half
        ax[1, 1].set_title('$t_{1/2}$')
        ax[1, 1].scatter(real_time(x_t_half_np, sec_per_frame), y_t_half_np, c='red', marker='x', s=s)

        # Formatting
        fig.suptitle(f'{roi}\n{prominence = }\n{rel_height = }')
        fig.supxlabel('Time / s')
        fig.supylabel('Normalised fluoresence intensity / a.u.')
        plt.tight_layout()
        sav_loc = output_dir + roi + '-t_half_validation.' + ext
        plt.savefig(sav_loc)
        if show:
            plt.show()
        plt.close()
