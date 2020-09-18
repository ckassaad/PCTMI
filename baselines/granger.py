"""
Granger Causality in Python
The algorithm is detailed here:
    https://pdfs.semanticscholar.org/8da4/f6a776d99ab99b31d5191bc773cc0473d34f.pdf
Date: Jan 2019
Author: Karim Assaad, karimassaad3@gmail.com
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.stats import f, levene
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
import time

from statsmodels.tsa.stattools import grangercausalitytests
from tools.functions import TsModel

from tools.functions import residuals

class Granger:
    def __init__(self, x, p=5, show=True):
        self.p = p
        self.d = x.shape[1]
        self.names = list(x.columns.values)
        # self.pa = {self.names[i]: self.names.copy() for i in range(len(self.names))}
        self.pa = {self.names[i]: [self.names[i]] for i in range(len(self.names))}

        min_max_scaler = StandardScaler()
        x_scaled = min_max_scaler.fit_transform(x.values)
        self.X = pd.DataFrame(x_scaled, columns=self.names)

        if show:
            for name in self.names:
                print("==================================")
                print("ADF test for "+str(name))
                print("==================================")
                result = adfuller(self.X[name])
                print('ADF Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('Critical Values:')
                # print(result)
                for key, value in result[4].items():
                    print('\t%s: %.3f' % (key, value))
            for name in self.names:
                print("==================================")
                print("KPSS test for "+str(name))
                print("==================================")
                result = kpss(self.X[name])
                print('KPSS Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('Critical Values:')
                # print(result)
                for key, value in result[3].items():
                    print('\t%s: %.3f' % (key, value))
            print()


    def predict(self, model, x):
        x_hat = pd.DataFrame(columns=x.columns.values)
        for t in range(x.shape[0] - self.p):
            temp = pd.DataFrame(model.forecast(x.values[t:(t + self.p)], 1), columns=list(x.columns.values))
            x_hat = x_hat.append(temp, ignore_index=True)
        return x_hat

    def f_test(self, var1, var2, m):
        f_ = np.divide(var1, var2)
        p_values = []
        for i in range(len(var1)):
            p_values.append(f.cdf(f_[i], m - 1, m - 1))
        return p_values

    def fit(self, test='F'):
        # try:
        model_full = VAR(self.X)
        model_full_fit = model_full.fit(maxlags=self.p, ic='aic')
        # except:
        #     model_full = TsModel(self.X)
        #     model_full_fit = []
        #     for name in self.X.columns:
        #         model_full_fit.append(model_full.fit(name))
        # print(model_full_fit.summary())

        # make prediction
        x_hat = self.predict(model_full_fit, self.X)

        # compute error
        err_full = residuals(x_hat.values, self.X.values[self.p:])
        var_full = list(np.var(err_full, axis=0))

        for j in range(self.d):
            x_temp = self.X.drop(columns=[self.names[j]])
            model_rest = VAR(x_temp)
            model_rest_fit = model_rest.fit(maxlags=self.p, ic='aic')

            # make prediction
            x_hat = self.predict(model_rest_fit, x_temp)

            # compute error
            err_rest = residuals(x_hat.values, x_temp.values[self.p:])
            var_rest = list(np.var(err_rest, axis=0))

            # F-test
            # err_full_rest = err_full.copy()
            # del err_full_rest[j]
            # test = np.less(err_rest,err_full_rest)
            # print(test)
            # for i in range(len(X_hat.columns.values)):
            #     if test[i]:
            #         self.pa[X_hat.columns.values[i]].remove(self.names[j])
            # print(self.pa)

            # F test (extremely sensitive to non-normality of X and Y)
            var_full_rest = var_full.copy()
            del var_full_rest[j]
            m = x_hat.shape[0]

            alpha = 0.05
            for i in range(len(x_hat.columns.values)):
                if test == 'F':
                    if var_rest[i] > var_full_rest[i]:
                        f_ = np.divide(var_rest[i], var_full_rest[i])
                    else:
                        f_ = np.divide(var_full_rest[i], var_rest[i])
                    print('F = ' + str(f_))

                    p_value = 1-f.cdf(f_, m-1, m-1)
                elif test == 'levene':
                    _, p_value = levene(err_rest[:, i], err_full[:, i])
                else:
                    print('The '+str(test)+' test is not supported')
                    exit()
                print('p value : '+str(p_value))
                if p_value < alpha:
                    # self.pa[X_hat.columns.values[i]].remove(self.names[j])
                    self.pa[x_hat.columns.values[i]].append(self.names[j])
        print(self.pa)

            # Test de Bartlett ou Test de Levene
            # bartlett()
        # print(var_rest)
        return self.pa


def granger_pairwise(data, alpha=0.05, p=5, test="ssr_chi2test"):
    X_train = pd.DataFrame(np.add(np.zeros([len(data.columns), len(data.columns)]), np.eye(len(data.columns))), columns=data.columns, index=data.columns)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=p, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(p)]
            min_p_value = np.min(p_values)
            if min_p_value < alpha:
                X_train.loc[r,c] = 2
                X_train.loc[c,r] = 1
    return X_train

def granger_adapted(data, p=5):
    # try:

    if data.shape[1]>2:
        G = Granger(data, p=p, show=False)
        res_dict = G.fit()
        print(res_dict)
        res_unit_array = np.zeros([data.shape[1], data.shape[1]])
        string_to_int = dict()
        i = 0
        for c in data.columns:
            string_to_int[c] = i
            i = i +1

        for k in res_dict.keys():
            k_int = string_to_int[k]
            for i in res_dict[k]:
                i_int = string_to_int[i]
                if k_int == i_int:
                    res_unit_array[i_int, k_int] = 1
                else:
                    res_unit_array[i_int,k_int] = 2
                    if res_unit_array[k_int,i_int] == 0:
                        res_unit_array[k_int,i_int] = 1
    else:
        res_unit_array = granger_pairwise(data, alpha=0.05, p=p, test="ssr_chi2test")
        res_unit_array = res_unit_array.values

    # except:
    #     print("granger pairwize")
    #     res_unit_array = granger_pairwise(data).values.transpose()
    return res_unit_array


if __name__ == "__main__":
    from data.sim_data import generate_v_structure, generate_fork, generate_mediator, generate_diamond, generate_fork_nl, generate_fork_nl_biglag

    data = generate_fork_nl(N=2000)

    start = time.time()
    # G = Granger(data)
    # res = G.fit()

    print(granger_adapted(data))
    end = time.time()


    # print(res)
    print("time: "+str(end-start))
