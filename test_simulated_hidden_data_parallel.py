import time
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed

from citmi import FCITMI
# from baselines.granger import granger_adapted
# from baselines.pcmci import pcmci_adapted
from baselines.tcdf import tcdf_adapted

from tools.evaluation import topology, precision_no, recall_no, f_score_no, hamming_distance, precision, recall, f_score

from data.sim_data import structure_with_2_hidden_var_generator, structure_with_2_hidden_var_generator_diff_sampling_rate


def run_on_data(i, method, structure, n_samples, non_linear, get_data, get_seed_constant, scale, verbose,
                save_scaled_data):
    random.seed(a=get_seed_constant[structure]*(i+1)/runs)
    np.random.seed((i+1))
    if verbose:
        print("############################## Run "+str(i)+" ##############################")
    data, ground_truth_unit, ground_truth_temporal = get_data[structure](N=n_samples, non_linear=non_linear)
    if verbose:
        print(ground_truth_unit)
    start = time.time()

    fci_output = ground_truth_unit.copy()
    if structure == "7ts_2h":
        fci_output[1, 2] = 3
        fci_output[2, 3] = 3
        fci_output[3, 2] = 3
        fci_output[3, 4] = 3
        fci_output[4, 3] = 3
        fci_output[4, 5] = 3
    elif structure == "hidden":
        fci_output[0, 1] = 3
        fci_output[1, 0] = 3
        fci_output[0, 2] = 3
        fci_output[2, 0] = 3
        fci_output[1, 2] = 3
        fci_output[2, 1] = 3

    if scale:
        data -= data.min()
        data /= data.max()

    if scale:
        scale_name = "scaled"
    else:
        scale_name = "unscaled"
    if save_scaled_data:
        data.to_csv("/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/data/simulated_ts_data/"+scale_name+"/"+str(structure)+"/data_"+str(i)+".csv")

    if method == "tsfci":
        path = "/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/experiments/causal_discovery/R_results/tsfci/"+str(structure)+"/res_data_"+str(i)+".csv"
        unit = pd.read_csv(path, index_col=0).values
    elif method == "tcdf":
        unit, temporal = tcdf_adapted(data, pivm=True)
    # elif method == "tcdfNoPIVM":
    #     unit, temporal = tcdf_adapted(data, pivm=False)
    elif method == "fcitmi":
        citmi = FCITMI(data, p_value=True, verbose=verbose)
        unit = citmi.fit().copy()
        unit2 = citmi.fit_gap_orientation().copy()
    else:
        unit = np.nan([ground_truth_unit.shape[0], ground_truth_unit.shape[0]])
        print("Error: method not found")
        exit(0)

    # evaluation
    if verbose:
        print(unit)
        if method == "fcitmi":
            print(unit2)
    topo = topology(unit, ground_truth_unit)
    ham = hamming_distance(unit, ground_truth_unit)

    pres_no = precision_no(unit, ground_truth_unit)
    rec_no = recall_no(unit, ground_truth_unit)
    fscore_no = f_score_no(unit, ground_truth_unit)

    pres = precision(unit, ground_truth_unit)
    rec = recall(unit, ground_truth_unit)
    fscore = f_score(unit, ground_truth_unit)

    ham_with_fci_out = hamming_distance(unit, fci_output)

    if method == "fcitmi":
        pres2 = precision(unit2, ground_truth_unit)
        rec2 = recall(unit2, ground_truth_unit)
        fscore2 = f_score(unit2, ground_truth_unit)
        ham_with_fci_out2 = hamming_distance(unit2, fci_output)

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
        print("hamm fci out: " + str(ham_with_fci_out))
        if method == "fcitmi":
            print("precision2: " + str(pres2))
            print("recall2: " + str(rec2))
            print("f-score2: " + str(fscore2))
            print("hamm fci out2: " + str(ham_with_fci_out2))
    if method == "fcitmi":
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, ham_with_fci_out, (end - start), pres2, rec2, \
               fscore2, ham_with_fci_out2
    else:
        return topo, pres_no, rec_no, fscore_no, ham, pres, rec, fscore, ham_with_fci_out, (end - start)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 5:
        print(len(sys.argv))
        method = sys.argv[1]    # fcitmi, tcdf, tsfci
        structure = sys.argv[2]
        n_samples = int(sys.argv[3])
        non_linear = bool(int(sys.argv[4])) # 0 for false and 1 for true
        scale = bool(int(sys.argv[5])) # 0 for false and 1 for true
        runs = int(sys.argv[6])
        num_processor = int(sys.argv[7]) # -1 for all
        verbose = bool(int(sys.argv[8]))
        print('Argument List:', str(sys.argv))
    else:
        print('Missing arguments so will take default arguments')
        method = "fcitmi"
        structure = "7ts_2h_dsr"
        n_samples = 1000
        non_linear = True   # 0 for false and 1 for true
        scale = False
        runs = 10
        num_processor = 1
        verbose = True
        print('Default Argument List:', str(method), str(structure), str(n_samples), str(non_linear), str(runs),
              num_processor)

    save_scaled_data = False

    get_data = {"7ts_2h": structure_with_2_hidden_var_generator, "7ts_2h_dsr": structure_with_2_hidden_var_generator_diff_sampling_rate}
    get_seed_constant = {"7ts_2h": 0.7, "7ts_2h_dsr": 0.7}

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
    ham_with_fci_out_list = results[:, 8]
    comput_time_list = results[:, 9]
    if method == "fcitmi":
        precision_list2 = results[:, 10]
        recall_list2 = results[:, 11]
        f_score_list2 = results[:, 12]
        ham_with_fci_out_list2 = results[:, 13]

    with open("./experiments/causal_discovery/results_hidden/"+str(method)+"_"+str(structure)+"_"+str(n_samples)+"_" +
              str(non_linear)+"_"+str(scale)+"_"+str(runs)+"_"+str(num_processor), "w+") as file:
        file.write("Summary Unit\n")
        file.write("Unit Topology: \n" + str(np.mean(topo_list)) + " +- " + str(np.std(topo_list)))
        file.write("\n")
        file.write("Unit Precision Non Oriented: \n" + str(np.mean(precision_no_list)) + " +- " + str(np.std(precision_no_list)))
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
        file.write("Hamming with FCI output: \n" + str(np.mean(ham_with_fci_out_list)) + " +- " + str(np.std(ham_with_fci_out_list)))
        file.write("\n")
        if method == "fcitmi":
            file.write("Unit Precision2: \n" + str(np.mean(precision_list2)) + " +- " + str(np.std(precision_list2)))
            file.write("\n")
            file.write("Unit Recall2: \n" + str(np.mean(recall_list2)) + " +- " + str(np.std(recall_list2)))
            file.write("\n")
            file.write("Unit F-Score2: \n" + str(np.mean(f_score_list2)) + " +- " + str(np.std(f_score_list2)))
            file.write("\n")
            file.write("Hamming with FCI output2: \n" + str(np.mean(ham_with_fci_out_list2)) + " +- " + str(
                np.std(ham_with_fci_out_list2)))
            file.write("\n")
        file.write("\n\nComputational Time: " + str(np.mean(comput_time_list)) + " +- " + str(np.std(comput_time_list)))

    if verbose:
        print("####################### Final Result #######################")
        print("Topology: "+str(np.mean(topo_list))+ " +- " + str(np.std(topo_list)))
        print("Precision Non Oriented: "+str(np.mean(precision_no_list))+ " +- " + str(np.std(precision_no_list)))
        print("Recall Non Oriented: "+str(np.mean(recall_no_list))+ " +- " + str(np.std(recall_no_list)))
        print("F1 Non Oriented: "+str(np.mean(f_score_no_list))+ " +- " + str(np.std(f_score_no_list)))
        print("Hamming distance: " + str(np.mean(ham_list)) + " +- " + str(np.std(ham_list)))
        print("Precision: " + str(np.mean(precision_list)) + " +- " + str(np.std(precision_list)))
        print("Recall: " + str(np.mean(recall_list)) + " +- " + str(np.std(recall_list)))
        print("F-Score: " + str(np.mean(f_score_list)) + " +- " + str(np.std(f_score_list)))
        print("Hamming with FCI output: " + str(np.mean(ham_with_fci_out_list)) + " +- " +
              str(np.std(ham_with_fci_out_list)))
        if method == "fcitmi":
            print("Precision2: " + str(np.mean(precision_list2)) + " +- " + str(np.std(precision_list2)))
            print("Recall2: " + str(np.mean(recall_list2)) + " +- " + str(np.std(recall_list2)))
            print("F-Score2: " + str(np.mean(f_score_list2)) + " +- " + str(np.std(f_score_list2)))
            print("Hamming with FCI output2: " + str(np.mean(ham_with_fci_out_list2)) + " +- " +
                  str(np.std(ham_with_fci_out_list2)))
