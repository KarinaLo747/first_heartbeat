import glob
import pathlib
import pandas as pd
from first_heartbeat import circles_analysis
from first_heartbeat.constants import manual_peak_find_csv


def main() -> None:

    with open(manual_peak_find_csv, 'w') as file:
        file.write('dir_path\n')

    csv_files = glob.glob('data/raw/*/*/*/*.csv')

    lst_of_results_dict = []

    for csv in csv_files:
        data_dir = str(pathlib.Path(csv).parent) + '/'
        results_dict: dict[str, any] = circles_analysis.run_analysis(data_dir=data_dir, filter_regex='Mean')
        lst_of_results_dict.append(results_dict)

    results_df = pd.DataFrame(lst_of_results_dict)
    results_df.to_csv('analysis_results.csv', index=False)


if __name__ == '__main__':
    main()