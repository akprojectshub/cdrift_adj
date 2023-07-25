import matplotlib.pyplot as plt
import pandas as pd
from cdrift import evaluation
from cdrift.utils.helpers import readCSV_Lists, convertToTimedelta, importLog
import numpy as np
from datetime import datetime
from statistics import mean, harmonic_mean, stdev
from scipy.stats import iqr
from typing import List
import seaborn as sns
import re
import os
from pathlib import Path




def calcAccuracy(df: pd.DataFrame, param_names: List[str], lag_window: int):
    """Calculates the Accuracy Metric for the given dataframe by grouping by the given parameters and calculating the mean accuracy

    Args:
        df (pd.DataFrame): The dataframe containing the results to be evaluated
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
        computed_precision_dicts[name], computed_recall_dicts[name], computed_accuracy_dicts[name] = calcAccuracy(a_df, used_parameters[name], lag_window)
        try:
            best_param = max(computed_accuracy_dicts[name], key=lambda x: computed_accuracy_dicts[name][x])
        except:
            best_param = 'na'
        accuracy_best_param[name] = best_param
        # accuracies[name] = max(computed_accuracy_dicts[name].values())
        accuracies[name] = computed_accuracy_dicts[name][best_param]
        if verbose:
            print(f"{name}: {accuracies[name]}")
    return (accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param)

def main():
    LAG_WINDOW = 200

    CSV_PATH = Path("ResultsCDLG", "algorithm_results_drift_1_D.csv")
    OUT_PATH = Path("ResultsCDLG", "algorithm_results_drift_1_D_evaluation.csv")


    df = readCSV_Lists(CSV_PATH)
    #df.copy()
    #print(df["Algorithm"].unique())
    #['Bose J', 'Bose WC', "Earth Mover's Distance", 'Process Graph Metrics', 'Martjushev ADWIN J', 'LCDD', 'Martjushev J', 'Zheng DBSCAN', 'Maaradji Runs']

    used_parameters = {
            "Bose J": ["Window Size", "SW Step Size"],
            "Bose WC": ["Window Size", "SW Step Size"],
            "Martjushev J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
            "Martjushev ADWIN J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
            #"Martjushev WC": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
            "Maaradji Runs": ["Window Size", "SW Step Size"],
            "Earth Mover's Distance": ["Window Size", "SW Step Size"],
            "Process Graph Metrics": ["Min Adaptive Window", "Max Adaptive Window", "P-Value"],
            "Zheng DBSCAN": ["MRID", "Epsilon"],
            "LCDD": ["Complete-Window Size", "Detection-Window Size", "Stable Period"]
        }

    df_noiseless = df["Log"]

    accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param = calculate_accuracy_metric_df(df, LAG_WINDOW, used_parameters, verbose=False)
    print(accuracies)
    print(accuracy_best_param)
    pd.DataFrame([{'Algorithm': name, 'Accuracy': accuracies[name]} for name in accuracies.keys()]).sort_values(by="Algorithm", ascending=True)


if __name__ == '__main__':
    main()

