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
    show: bool = False,
) -> None:
    """_summary_

    Args:
        data (pandas.DataFrame): _description_
        sec_per_frame (float): _description_
        output_dir (str): _description_
        title (str, optional): _description_. Defaults to None.
        xlim (list[float], optional): _description_. Defaults to None.
        ext (str, optional): _description_. Defaults to 'svg'.
        show (bool, optional): _description_. Defaults to False.
    """

    fig, ax = plt.subplots()

    for col in data:
        x: numpy.ndarray = data[col].index * sec_per_frame
        y: numpy.ndarray = data[col].values
        ax.plot(x, y, label=col)

    ax.set_title(title)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Fluoresence intensity / a.u.')

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        title += f'-xlim_{xlim[0]}_to_{xlim[1]}'

    plt.legend(loc='lower right')
    plt.tight_layout()
    save_loc = output_dir + title.lower().replace(' ', '_') + '-time_vs_fluoresence_intensity' + '.' + ext.lower()
    plt.savefig(save_loc)
    logger.info(f'Saved {save_loc}')
    if show:
        plt.show()
