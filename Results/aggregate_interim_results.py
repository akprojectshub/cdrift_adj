import os
import glob
import pandas as pd
import pathlib

def join_csv_files(folder_path):
    # Get all CSV file paths in the folder
    output_file = f'{folder_path}_agg.csv'  # Replace with the desired output file path
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Read and concatenate CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Write the combined data to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"CSV files joined and saved to {output_file}.")
    return combined_df

if __name__ == '__main__':
    # Usage example
    print()
    paths = []
    for root, dirs, files in os.walk(os.getcwd(), topdown=False):
        for name in dirs:
            paths.append(name)
    master_df = pd.DataFrame()
    #folder_path = Path('/path/to/csv_folder'  # Replace with the folder path containing your CSV files
    for path in paths:
        combined_df = join_csv_files(path)
        master_df = pd.concat([master_df, combined_df])
    master_df.to_csv('algorithm_results.csv', index=False)
