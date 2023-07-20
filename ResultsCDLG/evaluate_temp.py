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

LAG_WINDOW = 200
MIN_SUPPORT = 1 # Minimum support for latency calculation --> a parameter setting must have at least 3 instances where a true positive was detected before it can be deemed the "best parameter setting" for latency calculation

CSV_PATH = "algorithm_results.csv"
OUT_PATH = "Evaluation_Results"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


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


def calculate_accuracy_metric_df(dataframe, lag_window, verbose=True):
    computed_accuracy_dicts = dict()
    computed_precision_dicts = dict()
    computed_recall_dicts = dict()

    accuracy_best_param = dict()

    accuracies = dict()
    for name, a_df in dataframe.groupby(by="Algorithm"):
        computed_precision_dicts[name], computed_recall_dicts[name], computed_accuracy_dicts[name] = calcAccuracy(a_df,
                                                                                                                  used_parameters[
                                                                                                                      name],
                                                                                                                  lag_window)

        best_param = max(computed_accuracy_dicts[name], key=lambda x: computed_accuracy_dicts[name][x])

        accuracy_best_param[name] = best_param

        # accuracies[name] = max(computed_accuracy_dicts[name].values())
        accuracies[name] = computed_accuracy_dicts[name][best_param]
        if verbose:
            print(f"{name}: {accuracies[name]}")

    return (accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param)

def main():

    df = readCSV_Lists(CSV_PATH)
    df.copy()
    print(df["Algorithm"].unique())

    ['Martjushev ADWIN J' 'Process Graph Metrics' "Earth Mover's Distance"
     'LCDD' 'Maaradji Runs' 'Martjushev J' 'Bose J' 'Bose WC' 'Zheng DBSCAN']

    shorter_names = {
        "Martjushev ADWIN J": "ADWIN J",
        "Process Graph Metrics": "PGM",
        "Earth Mover's Distance": "EMD",
        "LCDD": "LCDD",
        "Maaradji Runs":
        "Martjushev J":
        "Bose J": "J-Measure",
        "Bose WC":  "Window Count",
        "Zheng DBSCAN":  "RINV"
    }

    shorter_names = {
        "Zheng DBSCAN": "RINV",
        "ProDrift": "ProDrift",
        "Bose J": ,
        "Bose WC": "Window Count",
        "Martjushev ADWIN WC": "ADWIN WC",



    shorter_names = {
        "Zheng DBSCAN": "RINV",
        "ProDrift": "ProDrift",
        "Bose J": "J-Measure",
        "Bose WC": "Window Count",
        "Earth Mover's Distance": "EMD",
        "Process Graph Metrics": "PGM",
        "Martjushev ADWIN J": "ADWIN J",
        "Martjushev ADWIN WC": "ADWIN WC",
        "LCDD": "LCDD"
    }
    df["Algorithm"] = df["Algorithm"].map(shorter_names)
    print(df["Algorithm"].unique())

    used_parameters = {
            "Bose J": ["Window Size", "SW Step Size"],
            "Bose WC": ["Window Size", "SW Step Size"],
            "Martjushev ADWIN J": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
            "Martjushev ADWIN WC": ["Min Adaptive Window", "Max Adaptive Window", "P-Value", "ADWIN Step Size"],
            "ProDrift": ["Window Size", "SW Step Size"],
            "Earth Mover's Distance": ["Window Size", "SW Step Size"],
            "Process Graph Metrics": ["Min Adaptive Window", "Max Adaptive Window", "P-Value"],
            "Zheng DBSCAN": ["MRID", "Epsilon"],
            "LCDD": ["Complete-Window Size", "Detection-Window Size", "Stable Period"]
        }

    used_parameters = {
        shorter_names[name]: used_parameters[name]
        for name in used_parameters.keys()
    }

    df_noiseless = df["Log"]

    accuracies, computed_accuracy_dicts, computed_precision_dicts, computed_recall_dicts, accuracy_best_param = calculate_accuracy_metric_df(df_noiseless, LAG_WINDOW, verbose=False)

    pd.DataFrame([{'Algorithm': name, 'Accuracy': accuracies[name]} for name in accuracies.keys()]).sort_values(by="Algorithm", ascending=True)


if __name__ == '__main__':
    main()

