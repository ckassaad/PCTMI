from baselines.tigramite.tigramite.independence_tests import ParCorr, CMIknn
from ctmi import tmi, ctmi, get_sampling_rate, window_representation, window_size, align_matrix, get_alpha
from tools.distances import cort
import numpy as np
import pandas as pd
import random

from data.sim_data import fork_generator, v_structure_generator, pair_generator, indep_pair_generator
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) > 3:
        print(len(sys.argv))
        structure = sys.argv[1]
        main_method = sys.argv[2]
        non_linear = bool(int(sys.argv[3]))  # 0 for false and 1 for true
        runs = int(sys.argv[4])
        print('Argument List:', str(sys.argv))

    else:
        print('Missing arguments so will take default arguments')
        structure = "fork"  # pair indep_pair fork v_structure
        main_method = "ctmi_nw"  # "tmi" or "tmi_nw" or "mi" or "corr" or "cort" or "ctmi" or "cmi" or "parcorr"
        non_linear = True
        runs = 10
        print('Default Argument List:', str(structure), str(non_linear), str(main_method), str(runs))






    if structure == "pair":
        col1 = 0
        col2 = 1
        col3 = None
        if main_method not in ["tmi", "tmi_nw", "mi", "corr", "cort", "cort*"]:
            print("method does not handle conditionals")
            exit(0)
    elif structure == "indep_pair":
        col1 = 0
        col2 = 1
        col3 = None
        if main_method not in ["tmi", "tmi_nw", "mi", "corr", "cort", "cort*"]:
            print("method does not handle conditionals")
            exit(0)
    elif structure == "fork":
        col1 = 1
        col2 = 2
        col3 = 0
        if main_method not in ["ctmi", "ctmi_nw", "cmi", "parcorr"]:
            print("method must handle conditionals")
            exit(0)
    elif structure == "v_structure":
        col1 = 0
        col2 = 1
        col3 = 2
        if main_method not in ["ctmi", "ctmi_nw", "cmi", "parcorr"]:
            print("method must handle conditionals")
            exit(0)
    else:
        col1 = None
        col2 = None
        col3 = None

    #######
    scale = False
    if scale:
        scale_name = "scaled"
    else:
        scale_name = "unscaled"

    save_data = True
    #######

    s = time.time()

    n_samples_list = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000,
                      4250, 4500, 4750, 5000]

    get_data = {"fork": fork_generator, "v_structure": v_structure_generator, "pair": pair_generator,
                "indep_pair": indep_pair_generator}

    get_seed_constant = {"fork": 0.1, "v_structure": 0.2, "pair": 0.8, "indep_pair": 0.8}

    for it in range(runs):
        random.seed(a=get_seed_constant[structure] * (it + 1) / runs)
        np.random.seed((it + 1))
        data_init, _, _ = get_data[structure](N=n_samples_list[-1], non_linear=non_linear)
        if scale:
            data_init -= data_init.min()
            data_init /= data_init.max()

        if save_data:
            data_init.to_csv(
                "/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/data/simulated_ts_data_for_independence_measures/" + scale_name + "/" + str(
                    structure) + "/data_" + str(it) + ".csv")
        print(data_init)

        # path = "/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/data/simulated_ts_data_for_independence_measures/" + scale_name + "/" + str(
        #     structure) + "/data_" + str(it) + ".csv"
        # data = pd.read_csv(path, index_col=0)

    output = []
    for n_samples in n_samples_list:
        result = []
        for it in range(runs):
            print("iteration: " + str(it))

            path = "/home/kassaad/Documents/Codes/Causal-Summary-Graph-time-series-different-sampling-rate/data/simulated_ts_data_for_independence_measures/" + scale_name + "/" + str(
                structure) + "/data_" + str(it) + ".csv"
            data = pd.read_csv(path, index_col=0)
            data = data.loc[:n_samples-1]
            if main_method == "mi":
                cmik = CMIknn()
                X = np.concatenate((data[data.columns[col1]].to_frame().values,
                                    data[data.columns[col2]].to_frame().values), axis=1)
                xyz = np.array([0] + [1])
                res = cmik.get_dependence_measure(X.T, xyz)
                print("mi: " + str(res))
                result.append(res)
            elif main_method == "corr":
                res = np.correlate(data[data.columns[col1]] - data[data.columns[col1]].mean(),
                                   data[data.columns[col2]] - data[data.columns[col2]].mean(), mode='full')
                res = res / (n_samples * data[data.columns[col1]].std() * data[data.columns[col2]].std())
                print("cross corr: " + str(max(res)))
                result.append(max(res))
            elif main_method == "cort":
                res = cort(data[data.columns[col1]].values, data[data.columns[col2]].values, simple=True)
                print("cort: " + str(res))
                result.append(abs(res))
            # elif main_method == "cort*":
            #     res = 1 - cort(data[data.columns[col1]].values, data[data.columns[col2]].values, simple=False)
            #     print("cort: " + str(1 - res))
            #     result.append(res)
            elif main_method == "tmi":
                # s_rs = []
                # s_rs_dict = dict()

                # for col in range(data.shape[1]):
                #     _, s_r = get_sampling_rate(data[data.columns[col]])
                #     s_rs.append(s_r)
                #     s_rs_dict[data.columns[col]] = s_r
                #
                # alpha = get_alpha(data)
                # lags = [window_size(data, alpha)] * data.shape[1]
                #
                # for col in range(data.shape[1]):
                #     data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                #                                                          windows_size=lags[col])
                #     _, s_r = get_sampling_rate(data[data.columns[col]])
                #     s_rs.append(s_r)
                #     s_rs_dict[data.columns[col]] = s_r
                #
                # am = align_matrix(data_dict, data.columns, s_rs_dict)

                data_dict = dict()
                lags = []
                sampling_rate = dict()
                for col in range(data.shape[1]):
                    _, s_r = get_sampling_rate(data[data.columns[col]])
                    sampling_rate[data.columns[col]] = s_r

                alpha = get_alpha(data)

                for col in range(data.shape[1]):
                    lags.append(window_size(data[data.columns[col]], alpha=alpha))
                    data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                                                                            windows_size=lags[col])

                am = align_matrix(data_dict, data.columns, sampling_rate)

                data_col1 = data_dict[data.columns[col1]]
                data_col2 = data_dict[data.columns[col2]]

                _, res = tmi(data_col1, data_col2, sampling_rate_tuple=(sampling_rate[data.columns[col1]], sampling_rate[data.columns[col2]]),
                          gamma=am[data.columns[col2]].loc[data.columns[col1]], p_value=False)
                print("tmi: " + str(res))
                result.append(res)
            elif main_method == "tmi_nw":
                # data_dict = dict()
                # lags = [1, 1, 1]
                # s_rs = []
                # s_rs_dict = dict()
                #
                # for col in range(data.shape[1]):
                #     data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                #                                                          windows_size=lags[col])
                #     _, s_r = get_sampling_rate(data[data.columns[col]])
                #     s_rs.append(s_r)
                #     s_rs_dict[data.columns[col]] = s_r
                #
                # alpha = get_alpha(data, s_rs_dict)
                # am = align_matrix(data_dict, data.columns, s_rs_dict)
                #
                # data_col1 = data_dict[data.columns[col1]]
                # data_col2 = data_dict[data.columns[col2]]
                #
                # res = tmi(data_col1, data_col2, sampling_rate_tuple=(s_rs[col1], s_rs[col2]),
                #           gamma=am[data.columns[col2]].loc[data.columns[col1]])

                data_dict = dict()
                lags = []
                sampling_rate = dict()
                for col in range(data.shape[1]):
                    _, s_r = get_sampling_rate(data[data.columns[col]])
                    sampling_rate[data.columns[col]] = s_r

                alpha = get_alpha(data)

                lags = [1, 1, 1]
                for col in range(data.shape[1]):
                    data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                                                                            windows_size=lags[col])

                am = align_matrix(data_dict, data.columns, sampling_rate)

                data_col1 = data_dict[data.columns[col1]]
                data_col2 = data_dict[data.columns[col2]]

                _, res = tmi(data_col1, data_col2, sampling_rate_tuple=(sampling_rate[data.columns[col1]], sampling_rate[data.columns[col2]]),
                          gamma=am[data.columns[col2]].loc[data.columns[col1]], p_value=False)
                print("tmi no window: " + str(res))
                result.append(res)
            elif main_method == "cmi":
                # order = 1
                # res = cmi(data[data.columns[col1]], data[data.columns[col2]], data[data.columns[col3]], order, order,
                #           order)
                cmik = CMIknn()
                X = np.concatenate((data[data.columns[col1]].to_frame().values,
                                    data[data.columns[col2]].to_frame().values,
                                    data[data.columns[col3]].to_frame().values), axis=1)
                xyz = np.array([0] + [1] + [2])
                res = cmik.get_dependence_measure(X.T, xyz)
                print("cmi: " + str(res))
                result.append(res)
            elif main_method == "parcorr":
                pc = ParCorr()
                X = np.concatenate((data[data.columns[col1]].to_frame().values,
                                    data[data.columns[col2]].to_frame().values,
                                    data[data.columns[col3]].to_frame().values), axis=1)
                xyz = np.array([0] + [1] + [2])
                res = pc.get_dependence_measure(X.T, xyz)
                print("parcorr: " + str(abs(res)))
                result.append(abs(res))
            elif main_method == "ctmi":
                data_dict = dict()
                s_rs = []
                s_rs_dict = dict()

                for col in range(data.shape[1]):
                    _, s_r = get_sampling_rate(data[data.columns[col]])
                    s_rs.append(s_r)
                    s_rs_dict[data.columns[col]] = s_r

                alpha = get_alpha(data)
                lags = []

                for col in range(data.shape[1]):
                    lags.append(window_size(data[data.columns[col]], alpha=alpha))
                    data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                                                                         windows_size=lags[col])

                am = align_matrix(data_dict, data.columns, s_rs_dict)

                data_col1 = data_dict[data.columns[col1]]
                data_col2 = data_dict[data.columns[col2]]
                data_col3 = data_dict[data.columns[col3]]

                sampling_rate_dict = s_rs_dict
                if structure == "fork":
                    mission = "ci"
                else:
                    mission = "cd"
                res = ctmi(data_col1, data_col2, {data.columns[col3]: data_col3},
                           data.columns[col1], data.columns[col2],
                           sampling_rate_dict, gamma_matrix=am, k=10, mission=mission)
                print("ctmi: " + str(res))
                result.append(res)
            elif main_method == "ctmi_nw":
                data_dict = dict()
                s_rs = []
                s_rs_dict = dict()

                for col in range(data.shape[1]):
                    _, s_r = get_sampling_rate(data[data.columns[col]])
                    s_rs.append(s_r)
                    s_rs_dict[data.columns[col]] = s_r

                alpha = get_alpha(data)
                lags = []

                lags = [1, 1, 1]
                for col in range(data.shape[1]):
                    data_dict[data.columns[col]] = window_representation(data[data.columns[col]],
                                                                         windows_size=lags[col])

                am = align_matrix(data_dict, data.columns, s_rs_dict)

                data_col1 = data_dict[data.columns[col1]]
                data_col2 = data_dict[data.columns[col2]]
                data_col3 = data_dict[data.columns[col3]]

                sampling_rate_dict = s_rs_dict
                if structure == "fork":
                    mission = "ci"
                else:
                    mission = "cd"
                res = ctmi(data_col1, data_col2, {data.columns[col3]: data_col3},
                           data.columns[col1], data.columns[col2],
                           sampling_rate_dict, gamma_matrix=am, k=10, mission=mission)
                print("ctmi_nw: " + str(res))
                result.append(res)

        print(result)
        print("result:")
        print("(" + str(n_samples) + ", " + str(np.mean(result)) + ") +- (" + str(np.std(result)) + ", " + str(
            np.std(result)) + ")")
        print("time: " + str(time.time() - s))
        output.append("(" + str(n_samples) + ", " + str(np.mean(result)) + ") +- (" + str(np.std(result)) + ", " + str(
            np.std(result)) + ")")

    for i in range(len(n_samples_list)):
        print(output[i])

    with open("./experiments/independence_measures/" + str(main_method) + "_" + str(structure) + "_" + str(non_linear),
              "w+") as file:
        for i in range(len(n_samples_list)):
            file.write(output[i]+"\n")
