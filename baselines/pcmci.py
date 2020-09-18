from baselines.tigramite.tigramite.pcmci import PCMCI
from baselines.tigramite.tigramite.independence_tests import ParCorr, CMIknn
from baselines.tigramite.tigramite import data_processing as pp
import numpy as np


def pcmci_adapted(data, tau_max=3, cond_ind_test="CMIknn", alpha=0.05):
    if cond_ind_test == "CMIknn":
        cond_ind_test = CMIknn()
    elif cond_ind_test == "ParCorr":
        cond_ind_test = ParCorr()

    data_tigramite = pp.DataFrame(data.values, var_names=data.columns)

    pcmci = PCMCI(
        dataframe=data_tigramite,
        cond_ind_test=cond_ind_test,
        verbosity=1)
    pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=alpha)

    res_dict = pcmci.all_parents
    res_unit_array = np.zeros([data.shape[1], data.shape[1]])

    for k in res_dict.keys():
        temp = res_dict[k]
        temp = np.unique([x[0] for x in temp])
        for i in temp:
            if k == i:
                res_unit_array[k,i] = 1
            else:
                if res_unit_array[k,i] == 0:
                    res_unit_array[k,i] = 1
                res_unit_array[i,k] = 2
    return res_unit_array, res_dict


if __name__ == "__main__":
    import time
    from data.sim_data import generate_v_structure, generate_fork, generate_mediator, generate_diamond, \
        generate_fork_nl, generate_fork_nl_biglag

    data = generate_fork_nl(N=100)

    start = time.time()

    print(pcmci_adapted(data, cond_ind_test="CMIknn"))

    end = time.time()
    print("time: " + str(end - start))