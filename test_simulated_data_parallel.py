import time
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed

from citmi import PCTMI
from baselines.granger import granger_adapted
from baselines.pcmci import pcmci_adapted
from baselines.tcdf import tcdf_adapted

from tools.evaluation import topology, precision_no, recall_no, f_score_no, hamming_distance, precision, recall, f_score

from data.sim_data import fork_generator, v_structure_generator, mediator_generator, \
    diamond_generator, mooij_7ts, fork_generator_diff_sampling_rate, \
    v_structure_generator_diff_sampling_rate, mediator_generator_diff_sampling_rate, \
    diamond_generator_diff_sampling_rate, complex_7_mooij_dsr, \
    mooij_7ts_reduced, mooij_7ts_reduced2, mooij_7ts_reduced3, seven_ts_generator, pair_generator, indep_pair_generator,\
    pair_generator_diff_sampling_rate, seven_ts_generator_diff_sampling_rate


def run_on_data(i, method, structure, n_samples, non_linear, get_data, get_seed_constant, scale, verbose,
                save_scaled_data):
    random.seed(a=get_seed_constant[structure]*(i+1)/runs)
    np.random.seed((i+1))
    if verbose:
        print("############################## Run "+str(i)+" ##############################")
    data, ground_truth_unit, ground_truth_temporal = get_data[structure](N=n_samples, non_linear=non_linear)
    if verbose:
        print(ground_truth_unit)
        print(ground_truth_temporal)
    start = time.time()

    if scale:
        data -= data.min()
        data /= data.max()

    if scale:
        scale_name = "scaled"
    else:
        scale_name = "unscaled"
    if save_scaled_data:
        data.to_csv("/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/data/simulated_ts_data/"+scale_name+"/"+str(structure)+"/data_"+str(i)+".csv")

    if method == "pcmciK":
        unit, temporal = pcmci_adapted(data, cond_ind_test="CMIknn", tau_max=5, alpha=0.05)
    elif method == "pcmciP":
        unit, temporal = pcmci_adapted(data, cond_ind_test="ParCorr", tau_max=5, alpha=0.05)
    elif method == "granger":
        unit = granger_adapted(data, p=22)
        temporal = None
    elif method == "timino":
        path = "/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/experiments/causal_discovery/R_results/timino/"+str(structure)+"/res_data_"+str(i)+".csv"
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
    else:
        unit = np.nan([ground_truth_unit.shape[0], ground_truth_unit.shape[0]])
        temporal = dict()
        print("Error: method not found")
        exit(0)

    # evaluation
    if verbose:
        print(unit)
        print(temporal)
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
        print("Computation time: "+str(end-start))
    if method == "pctmi":
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, (end - start), pres2, rec2, fscore2
    else:
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, (end - start)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 5:
        print(len(sys.argv))
        method = sys.argv[1]  # pctmi, pcmciP, pcmiK, timino, tcdf, tcdfNoPIVM, granger
        structure = sys.argv[2]
        n_samples = int(sys.argv[3])
        non_linear = bool(int(sys.argv[4]))  # 0 for false and 1 for true
        scale = bool(int(sys.argv[5]))  # 0 for false and 1 for true
        runs = int(sys.argv[6])
        num_processor = int(sys.argv[7])  # -1 for all
        verbose = bool(int(sys.argv[8]))
        print('Argument List:', str(sys.argv))
    else:
        print('Missing arguments so will take default arguments')
        method = "pctmi"
        structure = "dsr_7ts_0h"
        n_samples = 1000
        non_linear = True
        scale = False
        runs = 10
        num_processor = 1
        verbose = True
        print('Default Argument List:', str(method), str(structure), str(n_samples), str(non_linear), str(runs),
              num_processor)

    save_scaled_data = False

    get_data = {"fork": fork_generator, "v_structure": v_structure_generator, "diamond": diamond_generator,
                "mediator": mediator_generator, "7ts": mooij_7ts, "dsr_fork": fork_generator_diff_sampling_rate,
                "dsr_v_structure": v_structure_generator_diff_sampling_rate,
                "dsr_mediator": mediator_generator_diff_sampling_rate,
                "dsr_diamond": diamond_generator_diff_sampling_rate, "dsr_complex": complex_7_mooij_dsr,
                "red": mooij_7ts_reduced, "red2": mooij_7ts_reduced2, "red3": mooij_7ts_reduced3,
                "7ts_0h": seven_ts_generator, "pair": pair_generator, "indep_pair": indep_pair_generator,
                "dsr_7ts_0h": seven_ts_generator_diff_sampling_rate, "dsr_pair": pair_generator_diff_sampling_rate}
    get_seed_constant = {"fork": 0.1, "v_structure": 0.2, "diamond": 0.3, "hidden": 0.4, "7ts": 0.5, "cycle": 0.6,
                         "mediator": 0.7, "red": 0.5, "red2": 0.5, "red3": 0.5,
                         "dsr_fork": 0.1, "dsr_v_structure": 0.2, "dsr_mediator": 0.7, "dsr_diamond": 0.3,
                         "dsr_complex": 0.5, "dsr_cycle": 0.6, "7ts_0h": 0.5, "pair": 0.8, "indep_pair": 0.8,
                         "dsr_7ts_0h": 0.5, "dsr_pair": 0.8}

    results = Parallel(n_jobs=num_processor)(delayed(run_on_data)(i, method, structure, n_samples, non_linear,
                                                                  get_data, get_seed_constant, scale, verbose,
                                                                  save_scaled_data) for i in range(runs))

    results = np.array(results).reshape(runs, -1)
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

    with open("./experiments/causal_discovery/results/"+str(method)+"_"+str(structure)+"_"+str(n_samples)+"_" +
              str(non_linear)+"_"+str(scale)+"_"+str(runs)+"_"+str(num_processor), "w+") as file:
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
        print("Topology: "+str(np.mean(topo_list)) + " +- " + str(np.std(topo_list)))
        print("Precision Non Oriented: "+str(np.mean(precision_no_list)) + " +- " + str(np.std(precision_no_list)))
        print("Recall Non Oriented: "+str(np.mean(recall_no_list)) + " +- " + str(np.std(recall_no_list)))
        print("F1 Non Oriented: "+str(np.mean(f_score_no_list)) + " +- " + str(np.std(f_score_no_list)))
        print("Hamming distance: " + str(np.mean(ham_list)) + " +- " + str(np.std(ham_list)))
        print("Precision: " + str(np.mean(precision_list)) + " +- " + str(np.std(precision_list)))
        print("Recall: " + str(np.mean(recall_list)) + " +- " + str(np.std(recall_list)))
        print("F-Score: " + str(np.mean(f_score_list)) + " +- " + str(np.std(f_score_list)))
        if method == "pctmi":
            print("Precision2: " + str(np.mean(precision_list2)) + " +- " + str(np.std(precision_list2)))
            print("Recall2: " + str(np.mean(recall_list2)) + " +- " + str(np.std(recall_list2)))
            print("F-Score2: " + str(np.mean(f_score_list2)) + " +- " + str(np.std(f_score_list2)))
