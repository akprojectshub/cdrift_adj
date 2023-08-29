import pandas as pd
from cdrift import evaluation
from cdrift.utils.helpers import readCSV_Lists
import numpy as np
from statistics import harmonic_mean
from typing import List
from pathlib import Path


def calcAccuracy(df: pd.DataFrame, param_names: List[str], lag_window: int):
    """Calculates the Accuracy Metric for the given dataframe by grouping by the given parameters and calculating the mean accuracy

    Args:
        df (pd.DataFrame): The dataframe containing the Results to be evaluated
        param_names (List[str]): The names of the parameters of this approach
        lag_window (int): The lag window to be used for the evaluation to determine true positives and false positives

    Returns:
        _type_: _description_
    """

    f1s = dict()
    recalls = dict()
    precisions = dict()
    # Group by parameter values to calculate accuracy per parameter setting, over all logs
    for parameters, group in df.groupby(by=param_names):
        # Calculate Accuracy for this parameter setting
        ## --> F1-Score, but first collect all TP and FP
        tps = 0
        fps = 0
        positives = 0
        detected = 0

        # Collect TP FP, etc.
        for index, row in group.iterrows():
            actual_cp = row["Actual Changepoints for Log"]
            detected_cp = row["Detected Changepoints"]
            tp, fp = evaluation.getTP_FP(detected_cp, actual_cp, lag_window)
            tps += tp
            fps += fp
            positives += len(actual_cp)
            detected += len(detected_cp)

        try:
            precisions[parameters] = tps / detected
        except ZeroDivisionError:
            precisions[parameters] = np.NaN

        try:
            recalls[parameters] = tps / positives
        except ZeroDivisionError:
            recalls[parameters] = np.NaN

        f1s[parameters] = harmonic_mean(
            [precisions[parameters], recalls[parameters]])  # If either is nan, the harmonic mean is nan
    return (precisions, recalls, f1s)


def calculate_accuracy_metric_df(dataframe, lag_window, used_parameters, verbose=True):
    computed_accuracy_dicts = dict()
    computed_precision_dicts = dict()
    computed_recall_dicts = dict()

    accuracy_best_param = dict()

    accuracies = dict()
    for name, a_df in dataframe.groupby(by="Algorithm"):
        print(f"WIP: {name}")
        computed_precision_dicts[name], computed_recall_dicts[name], computed_accuracy_dicts[name] = calcAccuracy(a_df,
                                                                                                                  used_parameters[
                                                                                                                      name],
                                                                                                                  lag_window)
        try:
            best_param = max(computed_accuracy_dicts[name], key=lambda x: computed_accuracy_dicts[name][x])
        except:
            continue
        accuracy_best_param[name] = best_param
        # accuracies[name] = max(computed_accuracy_dicts[name].values())
        accuracies[name] = computed_accuracy_dicts[name][best_param]
        if verbose:
            print(f"{name}: {accuracies[name]}")
    return (accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param)


def main(CSV_PATH, LAG_WINDOW):


    df = readCSV_Lists(CSV_PATH)

    used_parameters = {
        "Bose J": ["Window Size", "SW Step Size"],
        "Bose WC": ["Window Size", "SW Step Size"],
        "Martjushev J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
        "Martjushev ADWIN J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
        # "Martjushev WC": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
        "Maaradji Runs": ["Window Size", "SW Step Size"],
        "Earth Mover's Distance": ["Window Size", "SW Step Size"],
        "Process Graph Metrics": ["Min Adaptive Window", "Max Adaptive Window", "P-Value"],
        "Zheng DBSCAN": ["MRID", "Epsilon"],
        "LCDD": ["Complete-Window Size", "Detection-Window Size", "Stable Period"]
    }

    f1_score_best, f1_score, computed_precision_dicts, computed_recall_dicts, accuracy_best_param = calculate_accuracy_metric_df(
        df, LAG_WINDOW, used_parameters, verbose=False)

    # Save best results
    f1_score_best_df = convert_and_save_dict_to_flat_file(f1_score_best)
    best_param_df = convert_and_save_dict_to_flat_file(accuracy_best_param)

    best_f1_and_parameter_df = pd.merge(f1_score_best_df, best_param_df, on=['Category', 'Parameter'])
    best_f1_and_parameter_df.rename(columns={'Value_x': 'Best_f1', 'Value_y': 'Best_config'}, inplace=True)

    file_name = CSV_PATH.parts[-1][:-4] + '_best_setting' + f"_{LAG_WINDOW}.csv"
    best_f1_and_parameter_df.to_csv(Path(CSV_PATH.parent, file_name), index=False)

    # Save all evaluation measures
    f1_score_df = convert_and_save_dict_to_flat_file(f1_score)
    precision_df = convert_and_save_dict_to_flat_file(computed_precision_dicts)
    recall_df = convert_and_save_dict_to_flat_file(computed_recall_dicts)

    evaluation_measure_and_config = pd.merge(f1_score_df, precision_df, on=['Category', 'Parameter'])
    evaluation_measure_and_config = pd.merge(evaluation_measure_and_config, recall_df, on=['Category', 'Parameter'])
    evaluation_measure_and_config.rename(columns={'Value_x': 'F1', 'Value_y': 'Precision', 'Value': 'Recall'}, inplace=True)

    file_name = CSV_PATH.parts[-1][:-4] + '_evaluation_measure' + f"_{LAG_WINDOW}.csv"
    evaluation_measure_and_config.to_csv(Path(CSV_PATH.parent, file_name), index=False)

    return None


def convert_and_save_dict_to_flat_file(data_dict):
    rows = []
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                row = (key, sub_key, sub_value)
                rows.append(row)
        else:
            row = (key, '', value)
            rows.append(row)

    df = pd.DataFrame(rows, columns=['Category', 'Parameter', 'Value'])

    return df




if __name__ == '__main__':
    LAG_WINDOW = 200
    CSV_PATH = Path('Results', 'set_A', "algorithm_results_v1.csv")
    #CSV_PATH = Path('Results', 'set_A', "algorithm_results_v2.csv")
    main(CSV_PATH, LAG_WINDOW)
