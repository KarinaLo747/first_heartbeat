import glob
import pathlib
from first_heartbeat import circles_analysis
from first_heartbeat.constants import manual_peak_find_csv


def main() -> None:

    with open(manual_peak_find_csv, 'w') as file:
        file.write('dir_path\n')

    csv_files = glob.glob('data/raw/*/*/*/*.csv')

    for csv in csv_files:
        data_dir = str(pathlib.Path(csv).parent) + '/'
        circles_analysis.run_analysis(data_dir=data_dir, filter_regex='Mean')


if __name__ == '__main__':
    main()