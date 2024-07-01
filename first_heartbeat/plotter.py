import numpy
import logging
import pandas
import matplotlib.pyplot as plt


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
    logger.info(f'Saved {save_loc}')
