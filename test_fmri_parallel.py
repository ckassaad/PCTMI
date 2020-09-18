import time
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from citmi import PCTMI, FCITMI
from joblib import Parallel, delayed


from tools.evaluation import topology, precision_no, recall_no, f_score_no, hamming_distance, precision, recall, f_score

from baselines.granger import granger_adapted
from baselines.pcmci import pcmci_adapted
from baselines.tcdf import tcdf_adapted


def unit_graph_ground_truth(three_col_format, d):
    # a graph is a two columns array: the first column correspond to the childs and the second correspond
    # to the direct parents
    ground_truth_unit = np.zeros([d, d])
    for i in range(three_col_format.shape[0]):
        c = int(three_col_format[i,0])
        e = int(three_col_format[i,1])
        if c ==e:
            ground_truth_unit[c, e] = 1
        else:
            ground_truth_unit[c, e] = 2
            if ground_truth_unit[e, c] == 0:
                ground_truth_unit[e, c] = 1
    return ground_truth_unit


def temporal_graph_ground_truth(three_col_format):
    ground_truth_temporal = {}
    for i in range(three_col_format.shape[0]):
        e = int(three_col_format[i, 1])
        ground_truth_temporal[e] = []

    for i in range(three_col_format.shape[0]):
        c = int(three_col_format[i, 0])
        e = int(three_col_format[i, 1])
        ground_truth_temporal[e].append((c, -int(three_col_format[i, 2])))
    return ground_truth_temporal


def run_on_data(i, method, files_input_name, scale, verbose):
    if verbose:
        print("############################## Run "+str(i)+" ##############################")
        # print("timeseries"+str(i+1))
        print(files_input_name[i])
    # file_input_name = "timeseries"+str(i+1)+".csv"
    file_input_name = files_input_name[i]
    data = pd.read_csv('./data/fMRI_processed_by_Nauta/returns/small_datasets/' + file_input_name, delimiter=',', header=None)
    idx = file_input_name.split('timeseries')[1].split('.csv')[0]
    # file_ground_truth_name = "sim"+str(i+1)+"_gt_processed"
    file_ground_truth_name = "sim"+idx+"_gt_processed"
    if file_ground_truth_name == "nocause":
        ground_truth_unit = np.zeros([data.shape[1], data.shape[1]])
        ground_truth_temporal = dict()
    else:
        ground_truth = np.loadtxt('./data/fMRI_processed_by_Nauta/' + file_ground_truth_name + '.csv', delimiter=',')
        ground_truth_unit = unit_graph_ground_truth(ground_truth, d=data.shape[1])
        ground_truth_temporal = temporal_graph_ground_truth(ground_truth)
    start = time.time()

    print(ground_truth_unit)

    if scale:
        data -= data.min()
        data /= data.max()

    if method == "pcmciK":
        unit, temporal = pcmci_adapted(data, cond_ind_test="CMIknn", tau_max=5, alpha=0.05)
    elif method == "pcmciP":
        unit, temporal = pcmci_adapted(data, cond_ind_test="ParCorr", tau_max=5, alpha=0.05)
    elif method == "granger":
        unit = granger_adapted(data, p=5)
        temporal = None
    elif method == "timino":
        path = "/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/experiments/causal_discovery/R_results/timino/fmri/res_"+files_input_name[i]
        unit = pd.read_csv(path, index_col=0).values
        temporal = None
    elif method == "tcdf":
        unit, temporal = tcdf_adapted(data, pivm=True)
    elif method == "tcdfNoPIVM":
        unit, temporal = tcdf_adapted(data, pivm=False)
    elif method == "pctmi":
        citmi = PCTMI(data, p_value=True, verbose=verbose)
        unit = citmi.fit().copy()
        unit2 = citmi.fit_gap_orientation().copy()
        print(unit)
        print(unit2)
        temporal = None
    elif method == "fcitmi":
        citmi = FCITMI(data, p_value=True, plot=False, verbose=verbose)
        unit = citmi.fit()
    else:
        unit = np.nan([ground_truth_unit.shape[0], ground_truth_unit.shape[0]])
        temporal = dict()
        print("Error: method not found")
        exit(0)

    # evaluation
    print(unit)
    if method == "pctmi":
        print(unit2)
    topo = topology(unit, ground_truth_unit)
    ham = hamming_distance(unit, ground_truth_unit)

    pres_no = precision_no(unit, ground_truth_unit)
    rec_no = recall_no(unit, ground_truth_unit)
    fscore_no = f_score_no(unit, ground_truth_unit)
    pres = precision(unit, ground_truth_unit)
    rec = recall(unit, ground_truth_unit)
    fscore = f_score(unit, ground_truth_unit)

    if method == "pctmi":
        pres2 = precision(unit2, ground_truth_unit)
        rec2 = recall(unit2, ground_truth_unit)
        fscore2 = f_score(unit2, ground_truth_unit)

    end = time.time()
    if verbose:
        print("topology: " + str(topo))
        print("hamming distance: " + str(ham))
        print("precision Non Oriented: " + str(pres_no))
        print("recall Non Oriented: " + str(rec_no))
        print("f-score Non Oriented: " + str(fscore_no))
        print("precision: " + str(pres))
        print("recall: " + str(rec))
        print("f-score: " + str(fscore))
        if method == "pctmi":
            print("precision2: " + str(pres2))
            print("recall2: " + str(rec2))
            print("f-score2: " + str(fscore2))

    if method == "pctmi":
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, (end - start), pres2, rec2, fscore2
    else:
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, (end - start)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        method = sys.argv[1]
        scale = bool(int(sys.argv[2]))  # 0 for false and 1 for true
        num_processor = int(sys.argv[3])  # -1 for all
        verbose = bool(int(sys.argv[4])) # 0 for false and 1 for true
        print('Argument List:', str(sys.argv))
    else:
        method = "pctmi"
        scale = False
        num_processor = 1
        verbose = True
        print('Default Argument List:', str(method), num_processor)

    path_input = './data/fMRI_processed_by_Nauta/returns/small_datasets'
    files_input_name = [f for f in listdir(path_input) if isfile(join(path_input, f)) and not f.startswith('.')]
    results = Parallel(n_jobs=num_processor)(delayed(run_on_data)(i, method, files_input_name, scale,
                                                                  verbose) for i in range(6, 12)) #len(files_input_name)

    results = np.array(results).reshape(6, -1)
    # results = np.array(results).reshape(len(files_input_name), -1)
    topo_list = results[:, 0]
    precision_no_list = results[:, 1]
    recall_no_list = results[:, 2]
    f_score_no_list = results[:, 3]
    ham_list = results[:, 4]
    precision_list = results[:, 5]
    recall_list = results[:, 6]
    f_score_list = results[:, 7]
    comput_time_list = results[:, 8]
    if method == "pctmi":
        precision_list2 = results[:, 9]
        recall_list2 = results[:, 10]
        f_score_list2 = results[:, 11]

    with open("./experiments/causal_discovery/results/" + str(method) + "_FMRI"+ "_"+str(scale), "w+") as file:
        file.write("Summary Unit\n")
        file.write("Unit Topology: \n" + str(np.mean(topo_list)) + " +- " + str(np.std(topo_list)))
        file.write("\n")
        file.write("Unit Precision Non Oriented: \n" + str(np.mean(precision_no_list)) + " +- " +
                   str(np.std(precision_no_list)))
        file.write("\n")
        file.write("Unit Recall Non Oriented: \n" + str(np.mean(recall_no_list)) + " +- " + str(np.std(recall_no_list)))
        file.write("\n")
        file.write("Unit F1 Non Oriented: \n" + str(np.mean(f_score_no_list)) + " +- " + str(np.std(f_score_no_list)))
        file.write("\n")
        file.write("Unit Hamming distance: \n" + str(np.mean(ham_list)) + " +- " + str(np.std(ham_list)))
        file.write("\n")
        file.write("Unit Precision: \n" + str(np.mean(precision_list)) + " +- " + str(np.std(precision_list)))
        file.write("\n")
        file.write("Unit Recall: \n" + str(np.mean(recall_list)) + " +- " + str(np.std(recall_list)))
        file.write("\n")
        file.write("Unit F-Score: \n" + str(np.mean(f_score_list)) + " +- " + str(np.std(f_score_list)))
        file.write("\n")
        if method == "pctmi":
            file.write("Unit Precision2: \n" + str(np.mean(precision_list2)) + " +- " + str(np.std(precision_list2)))
            file.write("\n")
            file.write("Unit Recall2: \n" + str(np.mean(recall_list2)) + " +- " + str(np.std(recall_list2)))
            file.write("\n")
            file.write("Unit F-Score2: \n" + str(np.mean(f_score_list2)) + " +- " + str(np.std(f_score_list2)))
            file.write("\n")

        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    if verbose:
        print("####################### Final Result #######################")
        print("Topology: " + str(np.mean(topo_list)) + " +- " + str(np.std(topo_list)))
        print(
            "Precision Non Oriented: " + str(np.mean(precision_no_list)) + " +- " + str(np.std(precision_no_list)))
        print("Recall Non Oriented: " + str(np.mean(recall_no_list)) + " +- " + str(np.std(recall_no_list)))
        print("F1 Non Oriented: " + str(np.mean(f_score_no_list)) + " +- " + str(np.std(f_score_no_list)))
        print("Hamming distance: " + str(np.mean(ham_list)) + " +- " + str(np.std(ham_list)))
        print("Precision: " + str(np.mean(precision_list)) + " +- " + str(np.std(precision_list)))
        print("Recall: " + str(np.mean(recall_list)) + " +- " + str(np.std(recall_list)))
        print("F-Score: " + str(np.mean(f_score_list)) + " +- " + str(np.std(f_score_list)))
        if method == "pctmi":
            print("Precision2: " + str(np.mean(precision_list2)) + " +- " + str(np.std(precision_list2)))
            print("Recall2: " + str(np.mean(recall_list2)) + " +- " + str(np.std(recall_list2)))
            print("F-Score2: " + str(np.mean(f_score_list2)) + " +- " + str(np.std(f_score_list2)))
        print("Computation time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))
