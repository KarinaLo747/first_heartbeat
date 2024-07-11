import glob
from first_heartbeat import circles_analysis
from first_heartbeat.constants import manual_peak_find_csv


def main() -> None:

    data_dirs = glob.glob('data/raw/*/*/*/')

    with open(manual_peak_find_csv, 'w') as file:
        pass

    for data_dir in data_dirs:
        circles_analysis.run_analysis(data_dir=data_dir, filter_regex='Mean')


if __name__ == '__main__':
    main()