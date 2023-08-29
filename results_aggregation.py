import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def process_directory(directory_path):
    all_results = []  # To accumulate DataFrames from all folders

    for root, _, files in os.walk(directory_path):
        folder_name = os.path.basename(root)
        results = []

        for file in tqdm(files, desc=f"Processing {folder_name}", unit="file"):
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                function_return = pd.read_csv(file_path)
                results.append(function_return)

        if results:
            df = pd.concat(results, ignore_index=True)
            all_results.append(df)  # Accumulate the DataFrame for each folder
            #output_path = Path(dir_input, f"results_{folder_name}.csv")
            #df.to_csv(output_path, index=False)
            #print(f"Results for {folder_name} saved to {output_path}")

    # Create a single flattened DataFrame containing data from all folders
    flattened_df = pd.concat(all_results, ignore_index=True)
    flattened_output_path = Path(dir_output, f"algorithm_results_{dir_input[-2:]}.csv")
    flattened_df.to_csv(flattened_output_path, index=False)
    print(f"Flattened results from all folders saved to {flattened_output_path}")

if __name__ == '__main__':
    dir_input = Path('Reproducibility_Intermediate_Results_v1')
    #dir_input = Path('Reproducibility_Intermediate_Results_v2')
    dir_output = Path('Results', 'set_A')
    process_directory(dir_input)
