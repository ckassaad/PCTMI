from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

if __name__ == "__main__":
    measures_list = ["Unit Topology:", "Unit Precision Non Oriented:", "Unit Recall Non Oriented:",
                     "Unit F1 Non Oriented:", "Unit Precision:", "Unit Recall:", "Unit F-Score:"]
    nb_measures = len(measures_list)
    path_input = './results'
    files_input_name = [f for f in listdir(path_input) if isfile(join(path_input, f)) and not f.startswith('.')]
    print(files_input_name)
    methods_list = []
    for s in files_input_name:
        methods_list.append(s.split("_", 1)[0])
    methods_list = np.unique(methods_list)
    nb_methods = len(methods_list)
    print(methods_list)

    datasets_list = []
    for s in files_input_name:
        datasets_list.append(s.split("_", 1)[1])
    datasets_list = np.unique(datasets_list)
    nb_datasets = len(datasets_list)
    print(datasets_list)
    # datasets_list = ["Finance"]

    results_table = pd.DataFrame(np.zeros([nb_measures, nb_methods]), columns=methods_list, index=measures_list)
    for dataset in datasets_list:
        for s in files_input_name:
            if dataset == s.split("_", 1)[1]:
                method = s.split("_", 1)[0]
                with open(str(path_input)+"/"+str(method)+"_"+str(dataset), "r") as f:
                    for line in f:
                        for measure in measures_list:
                            if measure in line:
                                nextLine = next(f)
                                nextLine = nextLine.replace("+-", "\pm")
                                nextLine = nextLine.replace("\n", "")
                                nextline_processed = nextLine.split(" ")
                                nextline_processed[0] = round(float(nextline_processed[0]), 3)
                                nextline_processed[2] = round(float(nextline_processed[2]), 3)
                                nextline_processed = '$'+' '.join([str(elem) for elem in nextline_processed])+'$'
                                results_table[method].loc[measure] = nextline_processed
        print(results_table)
        results_table.to_csv(r'./summary/'+str(dataset)+'.txt', sep='&')

