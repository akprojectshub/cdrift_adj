import enum
import sys
import argparse

# Do parsing as soon as possible to avoid large imports just to, e.g., show help menu
parser = argparse.ArgumentParser(description="Sudden Control Flow Concept Drift Detection Algorithm Evaluation")
parser.add_argument("-s", "--save-cores", type=int, required=False, help="How many cores NOT to use. The higher, the less cores are used; Default: 2", default=2)
parser.add_argument("-O", "--overwrite", required=False, help="If present and the output file already exists, it is overwritten. If not present, the execution is cancelled if the file already exists", action="store_true")
parser.add_argument("-o", "--output", required=False, type=str, default="testAll/", help="Specifies the output Directory")
parser.add_argument("-npp", "--no-parallel-progress", required=False, help="If present, the progress bars will all be on the same line, overwriting eachother.", action="store_true")
parser.add_argument("-sh", "--shuffle", required=False, help="If present, the list of tasks is shuffled. This means the approaches are not worked through one-by-one. Also makes processes overwriting eachother in output file less likely. Specific protection for this is coming soon.", action="store_true")
parser.add_argument("-lw", "--lag-window", required=False, type=int, default=200, help="Lag Window size for F1-Score calculation. Default: 200")
parser.add_argument("-l", "--logs", required=False, type=str, default="noiseless", help="Which set of event logs to use. Defaults to \"noiseless\"", choices=["noiseless", "noisy", "approaches"])

args = parser.parse_args(sys.argv[1:])

import math
from multiprocessing import Pool, RLock, freeze_support
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from cdrift.approaches import earthmover, bose, martjushev

#Maaradji
from cdrift.approaches import maaradji as runs

# Zheng
from cdrift.approaches.zheng import applyMultipleEps

#Process Graph CPD
from cdrift.approaches import process_graph_metrics as pm

# Helper functions and evaluation functions
from cdrift import evaluation
from cdrift.utils import helpers

#Misc
import os
from tqdm import tqdm
from datetime import datetime
from colorama import Fore
from tqdm import tqdm
from pathlib import Path

# From https://stackoverflow.com/a/17303428
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   MAGENTA = '\033[35m'

# Enum of approaches
class Approaches(enum.Enum):
    BOSE = "Bose"
    MARTJUSHEV = "Martjushev"
    EARTHMOVER = "Earthmover"
    MAARADJI = "Maaradji"
    PROCESS_GRAPHS = "ProcessGraph"
    ZHENG = "Zheng"

#TODO: Make this configuration accessible through arguments
DO_APPROACHES = {
    Approaches.BOSE: True,
    Approaches.MARTJUSHEV: True,
    Approaches.EARTHMOVER: True,
    Approaches.MAARADJI: True,
    Approaches.PROCESS_GRAPHS: True,
    Approaches.ZHENG: True
}

#################################
############ HELPERS ############
#################################

def calcDurationString(startTime, endTime):
    """
        Formats start and endtime to duration in hh:mm:ss format
    """
    elapsed_time = math.floor(endTime - startTime)
    return datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')

def calcDurFromSeconds(seconds):
    """
        Formats ellapsed seconds into hh:mm:ss format
    """
    seconds = math.floor(seconds)
    return datetime.strftime(datetime.utcfromtimestamp(seconds), '%H:%M:%S')

def plotPvals(pvals, changepoints, actual_changepoints, path, xlabel="", ylabel="", autoScale:bool=False):
    """
        Plots a series of p-values with detected and known change points and saves the figure
        args:
            - pvals
                List or array of p-values to be plotted
            - changepoints
                List of indices where change points were detected
            - actual_changepoints
                List of indices of actual change points
            - path
                The savepath of the generated image
            - xlabel
                Label of x axis
            - ylabel
                Label of y axis
            - autoScale
                Boolean whether y axis should autoscale by matplotlib (True) or be limited (0,max(pvals)+0.1) (False)
    """
    # Plotting Configuration
    fig = plt.figure(figsize=(10,4))
    plt.plot(pvals)
    # Not hardcoded 0-1 because of earthmovers distance (and +.1 so 1 is also drawn)
    if not autoScale:
        plt.ylim(0,max(pvals)+.1)
    for cp in changepoints:
        plt.axvline(x=cp, color='red', alpha=0.5)
    for actual_cp in actual_changepoints:
        plt.axvline(x=actual_cp, color='gray', alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{path}")
    plt.close()

#################################
##### Evaluation Functions ######
#################################

def testBose(filepath, WINDOW_SIZE, res_path:Path, F1_LAG, cp_locations, position=None):
    csv_name = "evaluation_results.csv"

    j_dur = 0
    wc_dur = 0

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]
    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size

    j_start = default_timer()
    pvals_j = bose.detectChange_JMeasure_KS(log, WINDOW_SIZE)
    cp_j = bose.visualInspection(pvals_j, WINDOW_SIZE)
    j_dur = default_timer() - j_start

    wc_start = default_timer()
    pvals_wc = bose.detectChange_WC_KS(log, WINDOW_SIZE)
    cp_wc = bose.visualInspection(pvals_wc, WINDOW_SIZE)
    wc_dur = default_timer() - wc_start

    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)

    # Save the results #
    plotPvals(pvals_j,cp_j,cp_locations, Path(res_path,f"{savepath}_J"), "Trace Number", "Mean P-Value for all Activity Pairs")
    plotPvals(pvals_wc,cp_wc,cp_locations, Path(res_path,f"{savepath}_WC"), "Trace Number", "Mean P-Value for all Activity Pairs")
    np.save(Path(res_path,f"npy/{savepath}_J"), pvals_j, allow_pickle=True)
    np.save(Path(res_path,f"npy/{savepath}_WC"), pvals_wc, allow_pickle=True)

    new_entry_j = {
        'Algorithm/Options':"Bose Average J", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_j,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_j, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_j, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_J
    }
    new_entry_wc = {
        'Algorithm/Options':"Bose Average WC", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_wc,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_wc, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_wc, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_WC
    }

    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append(new_entry_j, ignore_index=True)
    resDF = resDF.append(new_entry_wc, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

    return (Approaches.BOSE,[new_entry_j, new_entry_wc])

def testMartjushev(filepath, WINDOW_SIZE, res_path, F1_LAG, cp_locations, position=None):
    csv_name = "evaluation_results.csv"
    PVAL = 0.55
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]
    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size

    j_start = default_timer()
    rb_j_cp, rb_j_pvals = martjushev.detectChange_JMeasure_KS(log, WINDOW_SIZE, PVAL, return_pvalues=True, progressBarPos=position)
    j_dur = default_timer() - j_start

    wc_start = default_timer()
    rb_wc_cp, rb_wc_pvals = martjushev.detectChange_WindowCount_KS(log, WINDOW_SIZE, PVAL, return_pvalues=True, progressBarPos=position)
    wc_dur = default_timer() - wc_start
    
    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)


    # Save Results #    
    plotPvals(rb_j_pvals, rb_j_cp, cp_locations, Path(res_path,f"{savepath}_J_RecursiveBisection"), "Trace Number", "PValue")
    plotPvals(rb_wc_pvals, rb_wc_cp, cp_locations, Path(res_path,f"{savepath}_WC_RecursiveBisection"), "Trace Number", "PValue")
    np.save(Path(res_path,f"npy/{savepath}_J"), rb_j_pvals, allow_pickle=True)
    np.save(Path(res_path,f"npy/{savepath}_WC"), rb_wc_pvals, allow_pickle=True)

    ret = []
    new_entry_j = {
        'Algorithm/Options':f"Martjushev Recursive Bisection; Average J; p={PVAL}", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': rb_j_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=rb_j_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_j_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_J
    }
    new_entry_wc = {
        'Algorithm/Options':f"Martjushev Recursive Bisection; Average WC; p={PVAL}", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': rb_wc_cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=rb_wc_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_wc_cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr_WC
    }

    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append(new_entry_j, ignore_index=True)
    resDF = resDF.append(new_entry_wc, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

    return (Approaches.MARTJUSHEV, [new_entry_j, new_entry_wc])

def testEarthMover(filepath, WINDOW_SIZE, res_path, F1_LAG, cp_locations, position):
    LINE_NR = position
    csv_name = "evaluation_results.csv"

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size
    
    startTime = default_timer()

    # Earth Mover's Distance
    traces = earthmover.extractTraces(log)
    em_dists = earthmover.calculateDistSeries(traces, WINDOW_SIZE, progressBar_pos=LINE_NR)

    cp_em = earthmover.visualInspection(em_dists, WINDOW_SIZE)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    res_folder = Path(res_path, "EarthMover")

    plotPvals(em_dists,cp_em,cp_locations,Path(res_path,f"{savepath}_EarthMover_Distances"), autoScale=True)
    np.save(Path(res_path,f"npy/{savepath}_EarthMover_Distances"), em_dists, allow_pickle=True)
    resDF = pd.read_csv(Path(res_path,csv_name))

    new_entry = {
        'Algorithm/Options':"Earth Mover's Distance", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_em,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_em, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_em, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }
    
    resDF = resDF.append(new_entry, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

    return (Approaches.EARTHMOVER,[new_entry])

def testMaaradji(filepath, WINDOW_SIZE, res_path, F1_LAG, cp_locations, position):
    LINE_NR = position
    csv_name = "evaluation_results.csv"

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size

    startTime = default_timer()

    cp_runs, chis_runs = runs.detectChangepoints(log,WINDOW_SIZE, pvalue=0.05, return_pvalues=True, progressBar_pos=LINE_NR)
    actual_cp = cp_locations

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    res_folder = Path(res_path, "Maaradji")

    plotPvals(chis_runs, cp_runs, actual_cp, Path(res_path,f"{savepath}_P_Runs"), "Trace Number", "Chi P-Value")
    np.save(Path(res_path,f"npy/{savepath}_P_Runs"), chis_runs, allow_pickle=True)
    resDF = pd.read_csv(Path(res_path,csv_name))

    new_entry = {
        'Algorithm/Options':"Maaradji Runs", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_runs,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_runs, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_runs, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }
    resDF = resDF.append(new_entry, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)
    
    return (Approaches.MAARADJI,[new_entry])

def testGraphMetrics(filepath, WINDOW_SIZE, ADAP_MAX_WIN, res_path, F1_LAG, cp_locations, position=None):
    csv_name = "evaluation_results.csv"
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    cp = pm.detectChange(log, WINDOW_SIZE, ADAP_MAX_WIN, pvalue=0.05, progressBarPosition=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    resDF = pd.read_csv(Path(res_path,csv_name))

    new_entry = {
        'Algorithm/Options':"Process Graph Metrics", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Max Adaptive Window': ADAP_MAX_WIN,
        'Detected Changepoints': cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr
    }
    resDF = resDF.append(new_entry, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

    return (Approaches.PROCESS_GRAPHS,[new_entry])

def testZhengDBSCAN(filepath, mrid, epsList, res_path, F1_LAG, cp_locations, position):
    # candidateCPDetection is independent of eps, so we can calculate the candidates once and use them for multiple eps!
    csv_name = "evaluation_results.csv"
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()
    
    # CPD #
    cps = applyMultipleEps(log, mrid=mrid, epsList=epsList, progressPos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    resDF = pd.read_csv(Path(res_path,csv_name))

    ret = []
    for eps in epsList:
        cp = cps[eps]

        new_entry = {
            'Algorithm/Options':f"Zheng DBSCAN", 
            'Log': logname,
            'MRID': mrid,
            'Epsilon': eps,
            'Detected Changepoints': cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr
        }
        resDF = resDF.append(new_entry, ignore_index=True)
        ret.append(new_entry)
    resDF.to_csv(Path(res_path,csv_name), index=False)
    return (Approaches.ZHENG,ret)

def testSomething(idx:int, vals:int):
    """Wrapper for testing functions, as for the multiprocessing pool, one can only use one function, not multiple

    Args:
        idx (int): Position-Index for the progress bar of the evaluation
        vals (Tuple[str,List]): Tuple of name of the approach, and its parameter values
    """
    name, arguments = vals

    if args.no_parallel_progress:
        idx = None

    if name == Approaches.BOSE:
        return testBose(*arguments, position=idx)
    elif name == Approaches.MARTJUSHEV:
        return testMartjushev(*arguments, position=idx)
    elif name == Approaches.EARTHMOVER:
        return testEarthMover(*arguments, position=idx)
    elif name == Approaches.MAARADJI:
        return testMaaradji(*arguments, position=idx)
    elif name == Approaches.PROCESS_GRAPHS:
        return testGraphMetrics(*arguments, position=idx)
    elif name == Approaches.ZHENG:
        return testZhengDBSCAN(*arguments, position=idx)


def init_dir(results_path, csv_name="evaluation_results.csv"):
    APPROACHES = [approach for approach, do_approach in DO_APPROACHES.items() if do_approach] # The approaches that are enabled
    paths = {
        approach: Path(results_path, approach.value, csv_name)
        for approach in APPROACHES
    }


    # Create the directories
    try:
        for approach in APPROACHES:
            if DO_APPROACHES[approach]:
                path = Path(paths[approach].parent)
                if approach in [Approaches.BOSE, Approaches.MARTJUSHEV, Approaches.EARTHMOVER, Approaches.MAARADJI]:
                    # If the approach also has some resulting time series (pvalues, emd, ...) that we want to save
                    path = Path(path, "npy")
                os.makedirs(Path(path), exist_ok=args.overwrite)
    except FileExistsError:
        print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: One of the output folders already exists in the output directory. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")
        sys.exit()

    # Create the CSV Files to write to
    def try_save_df(df:pd.DataFrame, path:Path):
        """Save the Dataframe to the given path, raise an error if the file already exists and --overwrite is not set"""
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"The File {path} already exists")
        else:
            df.to_csv(path,index=False)

    approach_parameter_names = {
        Approaches.BOSE: ["Window Size"],
        Approaches.MARTJUSHEV: ["Window Size"],
        Approaches.MAARADJI: ["Window Size"],
        Approaches.EARTHMOVER: ["Window Size"],
        Approaches.PROCESS_GRAPHS: ["Window Size", "Max Adaptive Window"],
        Approaches.ZHENG: ["MRID", "Epsilon"]
    }
    try:
        for approach in APPROACHES:
            results = pd.DataFrame(
                columns=['Algorithm/Options', 'Log'] + approach_parameter_names[approach] + ['Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Average Lag', 'Duration']
            )
            try_save_df(results, paths[approach])
    except FileExistsError:
        print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: One of the Output CSVs already exists. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")


def main():
    #Evaluation Parameters
    F1_LAG = args.lag_window

    # Directory for Logs to test
    logPaths_Changepoints = None
    if args.logs.lower() == "noiseless":
        root_dir = Path("Sample Logs","Noiseless")
        # Get all files in the directory
        logPaths_Changepoints = [
            (item.as_posix(), [999,1999]) # Path to the event log, and the change point locations in it
            for item in root_dir.iterdir()
            if item.is_file() and item.suffixes in [[".xes"], [".xes", ".gz"]] # Only work with XES files (.xes) and compressed XES Files (.xes.gz)
        ]
    elif args.logs.lower() == "noisy":
        root_dir = Path("Sample Logs","Noiseful")
        # Get all files in the directory
        logPaths_Changepoints = [
            (item.as_posix(), [999,1999]) # Path to the event log, and the change point locations in it
            for item in root_dir.iterdir()
            if item.is_file() and item.suffixes in [[".xes"], [".xes", ".gz"]] # Only work with XES files (.xes) and compressed XES Files (.xes.gz)
        ]
    elif args.logs.lower() == "approaches":
        root_dir = Path("Sample Logs","Misc Approaches")
        # Here a manual mapping from log to change point is necessary because they are always at different locations
        logPaths_Changepoints = [
            # Caution: Counting begins at 0
            ("Sample Logs/Misc Approaches/cpnToolsSimulationLog.xes.gz", [1199, 2399, 3599, 4799]),# A change every 1200 cases, 6000 cases in total (Skipping 5999 because a change on the last case doesnt make sense)
            ("Sample Logs/Misc Approaches/log_long_term_dep.xes.gz", [1999])# Change at the 2000th case
        ]


    RESULTS_PATH = Path(args.output)

    init_dir(RESULTS_PATH)
    
    RESULT_PATHS = {
        approach: Path(RESULTS_PATH, approach.value)
        for approach in Approaches
    }

    # Parameter Settings #
    # Window Sizes that we test
    windowSizes    = [100, 200, 300, 400, 500,  600        ]
    maxWindowSizes = [200, 400, 600, 800, 1000, 1200       ] 
    
    # Special Parameters for the approach by Zheng et al.
    mrids = [100,250,500]
    eps_modifiers = [0.1,0.2,0.3] # use x * mrid as epsilon, as the paper suggests
    eps_mrid_pairs = [
        (mrid,[mEps*mrid  for mEps in eps_modifiers]) 
        for mrid in mrids
    ]

    bose_args         =  [(path, winSize,               RESULT_PATHS[Approaches.BOSE]              , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes                                  ]
    martjushev_args   =  [(path, winSize,               RESULT_PATHS[Approaches.MARTJUSHEV]        , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes                                  ]
    em_args           =  [(path, winSize,               RESULT_PATHS[Approaches.EARTHMOVER]        , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes                                  ]
    maaradji_args     =  [(path, winSize,               RESULT_PATHS[Approaches.MAARADJI]          , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize             in windowSizes                                  ]
    pgraph_args       =  [(path, winSize, adapMaxWin,   RESULT_PATHS[Approaches.PROCESS_GRAPHS]    , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for winSize, adapMaxWin in zip(windowSizes, maxWindowSizes)             ]
    zhengDBSCAN_args  =  [(path, mrid,    epsList,      RESULT_PATHS[Approaches.ZHENG]             , F1_LAG, cp_locations)     for path, cp_locations in logPaths_Changepoints for mrid,epsList        in eps_mrid_pairs                               ]

    arguments = ( [] # Empty list here so i can just comment out ones i dont want to do
        + ([ (Approaches.ZHENG, args)            for args in zhengDBSCAN_args ] if DO_APPROACHES[Approaches.ZHENG          ]         else [])
        + ([ (Approaches.MAARADJI, args)         for args in maaradji_args    ] if DO_APPROACHES[Approaches.MAARADJI       ]         else [])
        + ([ (Approaches.PROCESS_GRAPHS, args)   for args in pgraph_args      ] if DO_APPROACHES[Approaches.PROCESS_GRAPHS ]         else [])
        + ([ (Approaches.EARTHMOVER, args)       for args in em_args          ] if DO_APPROACHES[Approaches.EARTHMOVER     ]         else [])
        + ([ (Approaches.BOSE, args)             for args in bose_args        ] if DO_APPROACHES[Approaches.BOSE           ]         else [])
        + ([ (Approaches.MARTJUSHEV, args)       for args in martjushev_args  ] if DO_APPROACHES[Approaches.MARTJUSHEV     ]         else [])
    )

    if args.shuffle:
        np.random.shuffle(arguments)

    time_start = default_timer()

    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    results = None
    with Pool(os.cpu_count()-args.save_cores,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        results = p.starmap(testSomething, enumerate(arguments))
    elapsed_time = math.floor(default_timer() - time_start)
    # Write instead of print because of progress bars (although it shouldnt be a problem because they are all done)
    elapsed_formatted = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
    tqdm.write(f"{Fore.GREEN}The execution took {elapsed_formatted}{Fore.WHITE}")



    approaches = {name for name, _ in results}
    for approach in approaches:
        return_values = [return_value for name, return_value in results if name == approach]
        # Flatten the list, could do both in one, but that wouldnt look pretty
        return_values = [item for return_value in return_values for item in return_value]

        df = pd.DataFrame(return_values)
        df.to_csv(Path(RESULT_PATHS[approach], "evaluation_results_final.csv"), index=False)


if __name__ == '__main__':
    main()

# Example Run commands for each Log-Type
    # python .\testAll_MP.py -s 4 -o .\testAll_ApproachLogs\ -sh -l approaches -O
    # python .\testAll_MP.py -s 4 -o .\testAll\ -sh -l noiseless -O
    # python .\testAll_MP.py -s 4 -o .\testAll_Noisy\ -sh -l noisy -O