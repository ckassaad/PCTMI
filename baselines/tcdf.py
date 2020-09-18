from baselines.TCDF_master import runTCDF as tcdf
import numpy as np
import time
import random

def tcdf_adapted(data, pivm=True, kernel_size = 4, hidden_layers = 1, nrepochs = 5000, learningrate = 0.01, optimizername = "Adam",
            dilation_c=4, loginterval = 500, seed = 1111, significance = 0.8, cuda = False):
    allcauses, alldelays, allreallosses, allscores, columns = tcdf.runTCDF(data, pivm=pivm, kernel_size=kernel_size,
                                                                           hidden_layers=hidden_layers,
                                                                           nrepochs=nrepochs,
                                                                           learningrate=learningrate,
                                                                           optimizername=optimizername,
                                                                           dilation_c=dilation_c,
                                                                           loginterval=loginterval, seed=seed,
                                                                           significance=significance, cuda=cuda)
    res_unit_array = np.zeros([data.shape[1], data.shape[1]])
    res_dict = {}
    for k in allcauses.keys():
        temp = allcauses[k]
        if temp != []:
            res_dict[k] = []
            for i in temp:
                if k == i:
                    res_unit_array[k,i] = 1
                else:
                    if res_unit_array[k,i] == 0:
                        res_unit_array[k,i] = 1
                    res_unit_array[i,k] = 2
                print((alldelays))
                print(alldelays[(k, i)])
                res_dict[k].append((i, -alldelays[(k,i)]))

    return res_unit_array, res_dict

if __name__ == "__main__":
    from data.sim_data import generate_v_structure, generate_fork, generate_mediator, generate_diamond, \
            generate_fork_nl, generate_fork_nl_biglag

    data = generate_diamond(N=5000)

    start = time.time()


    allcauses, alldelays, allreallosses, allscores, columns = tcdf.runTCDF(data)
    print(allcauses)
    print(alldelays)

    unit, tempo = tcdf_adapted(data)
    print(unit)
    print(tempo)


    end = time.time()
    print("time: " + str(end - start))