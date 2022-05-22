import sys
import argparse

parser = argparse.ArgumentParser(description="Sudden Control Flow Concept Drift Detection Algorithm Evaluation")
parser.add_argument("-s", "--save-cores", type=int, required=False, help="How many cores NOT to use. The higher, the less cores are used; Default: 2", default=2)
parser.add_argument("-O", "--overwrite", required=False, help="If present and the output file already exists, it is overwritten. If not present, the execution is cancelled if the file already exists", action="store_true")
parser.add_argument("-o", "--output", required=False, type=str, default="testAll/", help="Specifies the output Directory")
parser.add_argument("-npp", "--no-parallel-progress", required=False, help="If present, the progress bars will all be on the same line, overwriting eachother.", action="store_true")
parser.add_argument("-sh", "--shuffle", required=False, help="If present, the list of tasks is shuffled. This means the approaches are not worked through one-by-one. Also makes processes overwriting eachother in output file less likely. Specific protection for this is coming soon.", action="store_true")
# parser.add_argument("-n", "--noisy", required=False, help="If present, evaluation is performed on noisy event logs. Otherwise noiseless event logs", action="store_true")
parser.add_argument("-l", "--logs", required=False, type=str, default="noiseless", help="Which set of event logs to use. Defaults to \"noiseless\"", choices=["noiseless", "noisy", "approaches"])
# parser.add_argument("--skipBose", required=False, help="If present, Bose will not be calculated, and previous computations will not be overwritten", action="store_true")
# parser.add_argument("--skipMartjushev", required=False, help="If present, Martjushev will not be calculated, and previous computations will not be overwritten", action="store_true")
# parser.add_argument("--skipMaaradji", required=False, help="If present, Maaradji will not be calculated, and previous computations will not be overwritten", action="store_true")
# parser.add_argument("--skipEM", required=False, help="If present, Earthmover will not be calculated, and previous computations will not be overwritten", action="store_true")
# parser.add_argument("--skipPG", required=False, help="If present, Process Graph CPD will not be calculated, and previous computations will not be overwritten", action="store_true")
# parser.add_argument("--skipZheng", required=False, help="If present, the approach by Zheng will not be calculated, and previous computations will not be overwritten", action="store_true")

args = parser.parse_args(sys.argv[1:])

import math
from multiprocessing import Pool, RLock, freeze_support
from numbers import Number
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

#Earthmover
import earthmover

#Maaradji
import maaradji as runs

# Zheng
from zheng import applyMultipleEps

#Bose
import bose
import scipy.stats as stats
import martjushev
# from extraction import timeseries as ts
# from localization import algorithms as algs

#Process Graph CPD
import processGraphMetrics as pm

# Helper Functions
import helpers

#Evaluation functions
import evaluation

#Misc
import os
from os.path import exists
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


DO_BOSE = False
DO_MARTJUSHEV = False
DO_EARTHMOVER = False
DO_MAARADJI = False
DO_PROCESS_GRAPH = False
DO_ZHENG = False

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
    plt.rcParams['figure.figsize']=(10,4)

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

def findMinima(signal,trim:int=0):
    """
        Detects minima in a signal
    """
    # Only send the trimmed version into the peak-finding algorithm; Because the initial, and final zero-values are the default values, and no comparison was made there, so it doesn't count for the peak finding
    peaks= find_peaks(-signal[trim:len(signal)-trim], width=80, prominence=0.1)[0]
    return [x+trim for x in peaks] # Add the window that was lost from the beginning

def findMaxima(signal, trim:int=0):
    """
        Detects maxima in a signal
    """
    peaks= find_peaks(signal[trim:len(signal)-trim], width=80)[0]
    # return find_peaks(signal, width=80)[0] # Used for Earthmover Distance; The distances have a different nature than minima so prominence is ignored
    return [x+trim for x in peaks] # Correct the found indices, these indices count from the beginning of the trimmed version instead of from the beginning of the untrimmed version (which we want)

def testBose(filepath, WINDOW_SIZE, res_path:Path, F1_LAG, position=None):
    LINE_NR = position
    csv_name = "evaluation_results.csv"

    j_dur = 0
    wc_dur = 0
    # rc = ts.extractRelationTypeCount(logs)
    # re = ts.extractRelationEntropy(logs, rc=rc) # Use the previously calculated rc as opposed to calculating it anew in the RE Method

    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]
    # log = xes_importer.apply(filepath)
    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size
    #Extract the average pvalues of the test over all pairs of activities, as Bose et al. do it in the paper
    activities = helpers._getActivityNames(log)
    pvals_wc = np.zeros(len(log))
    pvals_j = np.zeros(len(log))
    progress_j_wc = helpers.makeProgressBar(pow(len(activities),2),"extracting average p-values for wc/j, activity pairs completed", position=LINE_NR)
    for act1 in activities:
        for act2 in activities:
            #For the window Size in the feature extraction, default to average tracelength in log

            j_start = default_timer()
            j = bose.extractJMeasure(log, act1, act2)
            j_dur += default_timer() - j_start

            wc_start = default_timer()
            wc = bose.extractWindowCount(log, act1, act2)
            wc_dur += default_timer()-wc_start

            j_start = default_timer()
            new_pvals_j = bose.KSTest_2Sample_SlidingWindow(j,WINDOW_SIZE)
            j_dur += default_timer() - j_start

            wc_start = default_timer()
            new_pvals_wc = bose.KSTest_2Sample_SlidingWindow(wc,WINDOW_SIZE)
            wc_dur += default_timer()-wc_start

            #_, new_pvals = algs.MannWhitney_U(j, WINDOW_SIZE, 0.05, return_pvalues=True)
            pvals_j += new_pvals_j
            pvals_wc += new_pvals_wc
            progress_j_wc.update()
    pvals_j = pvals_j / pow(len(activities),2)
    pvals_wc = pvals_wc / pow(len(activities),2)

    ## Visual Inspection
    # cp_j = argrelmin(pvals_j, order=250)[0]
    # cp_wc = argrelmin(pvals_wc, order=250)[0]
    cp_j = findMinima(pvals_j, WINDOW_SIZE)
    cp_wc = findMinima(pvals_wc, WINDOW_SIZE)

    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)
    # Save the results #

    plotPvals(pvals_j,cp_j,[999,1999], Path(res_path,f"{savepath}_J"), "Trace Number", "Mean P-Value for all Activity Pairs")
    plotPvals(pvals_wc,cp_wc,[999,1999], Path(res_path,f"{savepath}_WC"), "Trace Number", "Mean P-Value for all Activity Pairs")
    np.save(Path(res_path,f"npy/{savepath}_J"), pvals_j, allow_pickle=True)
    np.save(Path(res_path,f"npy/{savepath}_WC"), pvals_wc, allow_pickle=True)

    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append({
        'Algorithm/Options':"Bose Average J", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_j,
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=cp_j, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr_J
    }, ignore_index=True)

    resDF = resDF.append({
        'Algorithm/Options':"Bose Average WC", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_wc,
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=cp_wc, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr_WC
    }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)
    progress_j_wc.close()

    # RE and RC
    # We, just as in the Paper by Bose et al., split the log into sublogs with 50 traces each (resulting in 60 sublogs only)
    # logs = split.divideLogCaseGroups(log, 50)
    # rc = ts.extractRelationTypeCount(logs)
    # re = ts.extractRelationEntropy(logs, rc=rc)

    
def testMartjushev(filepath, WINDOW_SIZE, res_path, F1_LAG, position=None):
    LINE_NR = position
    csv_name = "evaluation_results.csv"
    PVAL = 0.55 #0.65
    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]
    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size
    
    j_dur = 0
    wc_dur = 0

    activities = helpers._getActivityNames(log)
    progress_j_wc_rb = helpers.makeProgressBar(pow(len(activities),2),"extracting j/wc for recursive bisection algorithm, activity pairs completed ", position=LINE_NR)
    
    sig_j = np.zeros((pow(len(activities),2), len(log)), ) # Axes will be swapped soon so sig[:x] splits based on time
    sig_wc = np.zeros((pow(len(activities),2), len(log))) # Axes will be swapped soon so sig[:x] splits based on time
    i = 0
    for act1 in activities:
        for act2 in activities:
            j_start = default_timer()
            js = bose.extractJMeasure(log, act1, act2)
            sig_j[i] = js
            j_dur += default_timer() - j_start

            wc_start = default_timer()
            wcs = bose.extractWindowCount(log, act1, act2)
            sig_wc[i] = wcs
            wc_dur += default_timer() - wc_start

            progress_j_wc_rb.update()
            i += 1
    # Flip axes
    sig_j = np.swapaxes(sig_j, 0,1)
    sig_wc = np.swapaxes(sig_wc, 0,1)

    def _getPValue(res)->float:
        if isinstance(res,Number):
            return res
        else:
            try:
                return res.pvalue
            except:
                raise Exception("Statistical Test Result does not match criteria; Need either a float or an object with pvalue attribute")
    def _applyAvgPVal(window1, window2, testingFunc):
        pvals = []
        w1 = np.swapaxes(window1, 0,1)
        w2 = np.swapaxes(window2, 0,1)
        for i in range(len(w1)):
            pval = _getPValue(testingFunc(w1[i],w2[i]))
            pvals.append(pval)
        return np.mean(pvals)
    j_start = default_timer()
    rb_j_cp, rb_j_pvals = martjushev.statisticalTesting_RecursiveBisection(sig_j, WINDOW_SIZE, PVAL, lambda x,y:_applyAvgPVal(x,y,bose.StatTest.KS), return_pvalues=True)
    j_dur += default_timer() - j_start

    wc_start = default_timer()
    rb_wc_cp, rb_wc_pvals = martjushev.statisticalTesting_RecursiveBisection(sig_wc, WINDOW_SIZE, PVAL,lambda x,y:_applyAvgPVal(x,y,bose.StatTest.KS), return_pvalues=True)
    wc_dur += default_timer() - wc_start

    durStr_J = calcDurFromSeconds(j_dur)
    durStr_WC = calcDurFromSeconds(wc_dur)
    # Save Results #
        
    plotPvals(rb_j_pvals, rb_j_cp, [999,1999], Path(res_path,f"{savepath}_J_RecursiveBisection"), "Trace Number", "PValue")
    plotPvals(rb_wc_pvals, rb_wc_cp, [999,1999], Path(res_path,f"{savepath}_WC_RecursiveBisection"), "Trace Number", "PValue")
    np.save(Path(res_path,f"npy/{savepath}_J"), rb_j_pvals, allow_pickle=True)
    np.save(Path(res_path,f"npy/{savepath}_WC"), rb_wc_pvals, allow_pickle=True)

    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append({
        'Algorithm/Options':f"Martjushev Recursive Bisection; Average J; p={PVAL}", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': rb_j_cp,
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=rb_j_cp, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr_J
    }, ignore_index=True)
    resDF = resDF.append({
        'Algorithm/Options':f"Martjushev Recursive Bisection; Average WC; p={PVAL}", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': rb_wc_cp,
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=rb_wc_cp, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr_WC
    }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)
    progress_j_wc_rb.close()

def testEarthMover(filepath, WINDOW_SIZE, res_path, F1_LAG, position):
    LINE_NR = position
    csv_name = "evaluation_results.csv"

    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]

    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size
    
    startTime = default_timer()

    # Earth Mover's Distance
    traces = earthmover.extractTraces(log)
    em_dists = earthmover.calculateDistSeries(traces, WINDOW_SIZE, progressBar_pos=LINE_NR)

    # cp_em = argrelmax(em_dists, order=250)[0]
    cp_em = findMaxima(em_dists, WINDOW_SIZE)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    res_folder = Path(res_path, "EarthMover")

    plotPvals(em_dists,cp_em,[999,1999],Path(res_path,f"{savepath}_EarthMover_Distances"), autoScale=True)
    np.save(Path(res_path,f"npy/{savepath}_EarthMover_Distances"), em_dists, allow_pickle=True)
    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append({
        'Algorithm/Options':"Earth Mover's Distance", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_em, # Visual Inspection
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG,detected=cp_em, known=[999,1999], zero_division=np.NaN), # As visual inspection is required (for the time being)
        'Duration': durStr
    }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

def testMaaradji(filepath, WINDOW_SIZE, res_path, F1_LAG, position):
    LINE_NR = position
    csv_name = "evaluation_results.csv"

    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]

    savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size

    startTime = default_timer()

    cp_runs, chis_runs = runs.detectChangepoints(log,WINDOW_SIZE, pvalue=0.05, return_pvalues=True, progressBar_pos=LINE_NR)
    actual_cp = [999,1999]

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    res_folder = Path(res_path, "Maaradji")

    plotPvals(chis_runs, cp_runs, actual_cp, Path(res_path,f"{savepath}_P_Runs"), "Trace Number", "Chi P-Value")
    np.save(Path(res_path,f"npy/{savepath}_P_Runs"), chis_runs, allow_pickle=True)
    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append({
        'Algorithm/Options':"Maaradji Runs", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Detected Changepoints': cp_runs,
        'Actual Changepoints for Log': [999,1999],
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=cp_runs, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr
    }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

def testGraphMetrics(filepath, WINDOW_SIZE, ADAP_MAX_WIN, res_path, F1_LAG, position=None):
    csv_name = "evaluation_results.csv"
    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]

    # savepath = f"{logname}_W{WINDOW_SIZE}"# The file name without extension + the window size

    startTime = default_timer()

    cp = pm.detectChange(log, WINDOW_SIZE, ADAP_MAX_WIN, pvalue=0.05, progressBarPosition=position)
    actual_cp = [999,1999]

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    resDF = pd.read_csv(Path(res_path,csv_name))
    resDF = resDF.append({
        'Algorithm/Options':"Process Graph Metrics", 
        'Log': logname,
        'Window Size': WINDOW_SIZE,
        'Max Adaptive Window': ADAP_MAX_WIN,
        'Detected Changepoints': cp,
        'Actual Changepoints for Log': actual_cp,
        'F1-Score': evaluation.F1_Score(F1_LAG, detected=cp, known=[999,1999], zero_division=np.NaN),
        'Duration': durStr
    }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)

def testZhengDBSCAN(filepath, mrid, epsList, res_path, F1_LAG, position):
    # candidateCPDetection is independent of eps, so we can use the calculated candidates for multiple eps!
    csv_name = "evaluation_results.csv"
    log = helpers.importLogSilent(filepath)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()
    
    # CPD #
    cps = applyMultipleEps(log, mrid=mrid, epsList=epsList, progressPos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    resDF = pd.read_csv(Path(res_path,csv_name))
    for eps in epsList:
        cp = cps[eps]
        resDF = resDF.append({
            'Algorithm/Options':f"Zheng DBSCAN", 
            'Log': logname,
            'MRID': mrid,
            'Epsilon': eps,
            'Detected Changepoints': cp,
            'Actual Changepoints for Log': [999,1999],
            'F1-Score': evaluation.F1_Score(F1_LAG, detected=cp, known=[999,1999], zero_division=np.NaN),
            'Duration': durStr
        }, ignore_index=True)
    resDF.to_csv(Path(res_path,csv_name), index=False)


def testSomething(idx, vals):
    name, arguments = vals

    if args.no_parallel_progress:
        idx = None

    if name == "bose":
        testBose(*arguments, position=idx)
    elif name == "martjushev":
        testMartjushev(*arguments, position=idx)
    elif name == "earthmover":
        testEarthMover(*arguments, position=idx)
    elif name == "maaradji":
        testMaaradji(*arguments, position=idx)
    elif name == "pgraphmetrics":
        testGraphMetrics(*arguments, position=idx)
    elif name == "zhengDBSCAN":
        testZhengDBSCAN(*arguments, position=idx)


def init_dir(results_path):
    global DO_BOSE
    global DO_MARTJUSHEV
    global DO_EARTHMOVER
    global DO_MAARADJI
    global DO_PROCESS_GRAPH
    global DO_ZHENG
    # try:
    #     os.makedirs(results_path,exist_ok=args.overwrite)
    # except FileExistsError:
    #     print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: The folder {results_path} already exists. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")
    # Results_dir is made if any of the algs are done
    # png, npy folders creeation #
    try:
        if DO_BOSE:
            os.makedirs(Path(results_path,"Bose/npy"), exist_ok=args.overwrite)
        if DO_MARTJUSHEV:
            os.makedirs(Path(results_path,"Martjushev/npy"), exist_ok=args.overwrite)
        if DO_EARTHMOVER:
            os.makedirs(Path(results_path,"Earthmover/npy"), exist_ok=args.overwrite)
        if DO_MAARADJI:
            os.makedirs(Path(results_path,"Maaradji/npy"), exist_ok=args.overwrite)
        # os.makedirs(Path(results_path,"ProcessGraph/npy")) # It doesnt save any figures or similar yet.
        # os.makedirs(Path(results_path,"zheng/npy")) # Also no files generated. Cannot think of any to generate either, also very fast so no real reason to save if you can just calculate
    except FileExistsError:
        print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: One of the output folders already exists in the output directory. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")
        sys.exit()

    # Create CSV's
    def try_save_csv(csv, path:Path):
        if path.exists():
            raise FileExistsError(f"The File {path} already exists")
        else:
            csv.to_csv(path,index=False)
        pass
    csv_name = "evaluation_results.csv"
    paths = {
        "Bose":         Path(results_path,"Bose",csv_name),
        "Martjushev":   Path(results_path,"Martjushev",csv_name),
        "Maaradji":     Path(results_path,"Maaradji",csv_name),
        "Earthmover":   Path(results_path,"Earthmover",csv_name),
        "ProcessGraph": Path(results_path,"ProcessGraph",csv_name),
        "Zheng":        Path(results_path,"Zheng",csv_name)
    }
    try:
        if DO_ZHENG:
            os.makedirs(paths["Zheng"].parent, exist_ok=args.overwrite)
        if DO_PROCESS_GRAPH:
            os.makedirs(paths["ProcessGraph"].parent, exist_ok=args.overwrite)

    except FileExistsError:
        print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: One of the output folders already exists in the output directory. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")
        sys.exit()
    if args.overwrite:
        results = pd.DataFrame(
            columns=['Algorithm/Options', 'Log', 'Window Size', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
        )
        if DO_BOSE:
            results.to_csv(paths["Bose"], index=False)
        if DO_MARTJUSHEV:
            results.to_csv(paths["Martjushev"], index=False)
        if DO_MAARADJI:
            results.to_csv(paths["Maaradji"], index=False)
        if DO_EARTHMOVER:
            results.to_csv(paths["Earthmover"], index=False)
        if DO_PROCESS_GRAPH:
            results = pd.DataFrame(
                columns=['Algorithm/Options', 'Log', 'Window Size', 'Max Adaptive Window', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
            )
            results.to_csv(paths["ProcessGraph"], index=False)
        if DO_ZHENG:
            results = pd.DataFrame(
                columns=['Algorithm/Options', 'Log', 'MRID', 'Epsilon', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
            )
            results.to_csv(paths["Zheng"], index=False)
    else:
        try:
            results = pd.DataFrame(
                columns=['Algorithm/Options', 'Log', 'Window Size', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
            )
            if DO_BOSE:
                try_save_csv(results,paths["Bose"])
            if DO_MARTJUSHEV:
                try_save_csv(results,paths["Martjushev"])
            if DO_MAARADJI:
                try_save_csv(results,paths["Maaradji"])
            if DO_EARTHMOVER:
                try_save_csv(results,paths["Earthmover"])
            if DO_PROCESS_GRAPH:
                results = pd.DataFrame(
                    columns=['Algorithm/Options', 'Log', 'Window Size', 'Max Adaptive Window', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
                )
                try_save_csv(results, paths["ProcessGraph"])
            if DO_ZHENG:
                results = pd.DataFrame(
                    columns=['Algorithm/Options', 'Log', 'MRID', 'Epsilon', 'Detected Changepoints', 'Actual Changepoints for Log','F1-Score', 'Duration'],
                )
                try_save_csv(results, paths["Zheng"])
        except FileExistsError:
            print(f"{Fore.RED}{color.BOLD}Error{Fore.RESET}: One of the Output CSVs already exists. To overwrite, use --overwrite or -O. To specify a different output file, use --output or -o")


def main():
    global DO_BOSE
    global DO_MARTJUSHEV
    global DO_EARTHMOVER
    global DO_MAARADJI
    global DO_PROCESS_GRAPH
    global DO_ZHENG


    #TODO: Make this accessible from outside
    DO_BOSE = True
    DO_MARTJUSHEV = True
    DO_EARTHMOVER = True
    DO_MAARADJI = True
    DO_PROCESS_GRAPH = True
    DO_ZHENG = True


    #Evaluation Parameters
    F1_LAG = 200

    # Directory for Logs to test
    root_dir = None
    print(args.logs)
    print(args.logs.lower())
    if args.logs.lower() == "noiseless":
        root_dir = Path("Sample Logs","Noiseless")
    elif args.logs.lower() == "noisy":
        root_dir = Path("Sample Logs","Noiseful")
    elif args.logs.lower() == "approaches":
        root_dir = Path("Sample Logs","Misc Approaches")

    # Get all files in the directory
    logPaths = {
        item.as_posix()
        for item in root_dir.iterdir()
        if item.is_file() and item.suffixes in [[".xes"], [".xes", ".gz"]] # Only work with XES files (.xes) and compressed XES Files (.xes.gz)
    }


    # CSV_NAME = Path(args.output).name
    RESULTS_PATH = Path(args.output)

    init_dir(RESULTS_PATH)

    # Parameter Settings #
    # Window Sizes that we test
    windowSizes    = [100, 200, 300, 400, 500,  600        ]
    maxWindowSizes = [200, 400, 600, 800, 1000, 1200       ] 
    
    # valuePairs     = itertools.product(logPaths, windowSizes)
    
    # mrids = [100,200,300,400,500,750]
    mrids = [100,250,500]
    eps_modifiers = [0.1,0.2,0.3]
    eps_mrid_pairs = [
        (mrid,[mEps*mrid  for mEps in eps_modifiers]) 
        for mrid in mrids
    ]

    bose_args         =  [(path, winSize,               Path(RESULTS_PATH,"Bose"), F1_LAG)             for path in logPaths for winSize             in windowSizes                                  ]
    martjushev_args   =  [(path, winSize,               Path(RESULTS_PATH,"Martjushev"), F1_LAG)       for path in logPaths for winSize             in windowSizes                                  ]
    em_args           =  [(path, winSize,               Path(RESULTS_PATH, "Earthmover"), F1_LAG)      for path in logPaths for winSize             in windowSizes                                  ]
    maaradji_args     =  [(path, winSize,               Path(RESULTS_PATH, "Maaradji"), F1_LAG)        for path in logPaths for winSize             in windowSizes                                  ]
    pgraph_args       =  [(path, winSize, adapMaxWin,   Path(RESULTS_PATH, "ProcessGraph"), F1_LAG)    for path in logPaths for winSize, adapMaxWin in zip(windowSizes, maxWindowSizes)             ]
    zhengDBSCAN_args  =  [(path, mrid,    epsList,      Path(RESULTS_PATH, "Zheng"), F1_LAG)           for path in logPaths for mrid,epsList        in eps_mrid_pairs                               ]

    arguments = ( [] # Empty list here so i can just comment out ones i dont want to do
        + ([ ("zhengDBSCAN", args)   for args in zhengDBSCAN_args ] if DO_ZHENG          else [])
        + ([ ("maaradji", args)      for args in maaradji_args    ] if DO_MAARADJI       else [])
        + ([ ("pgraphmetrics", args) for args in pgraph_args      ] if DO_PROCESS_GRAPH  else [])
        + ([ ("earthmover", args)    for args in em_args          ] if DO_EARTHMOVER     else [])
        + ([ ("bose", args)          for args in bose_args        ] if DO_BOSE           else [])
        + ([ ("martjushev", args)    for args in martjushev_args  ] if DO_MARTJUSHEV     else [])
    )

    if args.shuffle:
        np.random.shuffle(arguments)

    time_start = default_timer()

    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    with Pool(os.cpu_count()-args.save_cores,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        p.starmap(testSomething, enumerate(arguments))
    elapsed_time = math.floor(default_timer() - time_start)
    # Write instead of print because of progress bars (although it shouldnt be a problem because they are all done)
    elapsed_formatted = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
    tqdm.write(f"{Fore.GREEN}The execution took {elapsed_formatted}{Fore.WHITE}")

if __name__ == '__main__':
    main()