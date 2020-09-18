"""
Date: Jan 2019
Author: Karim Assaad, karimassaad3@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lr
# from pygam import LinearGAM, s, f

from scipy.stats import pearsonr as cross_corr


#structural hamming distance

#hamming distance
##############################testtt ittt ##########
def hamming_distance(dag, dag_hat):
    Gtmp = (dag + dag_hat) % 2
    # Gtmp = Gtmp + np.transpose(Gtmp)
    # nrReversals = np.sum(Gtmp == 2) / 2
    # nrInclDel = np.sum(Gtmp == 1) / 2
    # hammingDis = nrReversals + nrInclDel

    # hammingDis = np.sum(abs(dag - dag_hat))
    # #    # correction: dist(-,.) = 1, not 2
    # hammingDis = hammingDis - 0.5*np.sum(dag * np.transpose(dag) * (1-dag_hat) * np.transpose(1-dag_hat) + dag_hat *
    #                                  np.transpose(dag_hat) * (1-dag) * np.transpose(1-dag))

    return np.sum(Gtmp)


def residuals(y, y_hat):
    """
    :param y: true values
    :param y_hat: predicted values
    :return: residuals
    """
    err = np.subtract(y_hat, y)
    return err


# better for prediction because it penelize less than BIC
def aic(y, y_hat, n):
    """
    :param y: true values
    :param y_hat: predicted values
    :param n: number of variables
    :return: AIC score
    """
    return 2*n - 2*np.log(np.sum(residuals(y, y_hat)**2))


# better for estimation (model selection) because it penelize more than AIC
def bic(y, y_hat, n, m):
    """
    :param y: true values
    :param y_hat: predicted values
    :param n: number of variables
    :param m: number of observations
    :return: BIC score
    """
    return m*np.log(np.sum(residuals(y, y_hat)**2)/m) + n*np.log(m)



def ts_var(ts, order=4):
    names = []
    for i in range(order):
        names.append(ts.name+"_"+str(i+1))
    new_mts = ts.values.reshape(-1,order)
    new_mts = pd.DataFrame(new_mts)
    return new_mts

# windows in representation (same function exist in tsMI)
def ts_order(ts, order=4):
    if (order==0) or (order==1):
        return ts.to_frame()
    else:
        new_mts = pd.DataFrame()
        for i in range(order):
            i_data = ts[i:(ts.shape[0]-order+i+1)].values
            new_mts.loc[:, ts.name+"_"+str(i+1)] = i_data
    return new_mts

def mts_order(mts, order=4):
    new_mts = pd.DataFrame()
    for i in range(order + 1):
        if i == order:
            i_data = mts[i:]
        else:
            i_data = mts[i:(-order + i)]
        if isinstance(mts, pd.DataFrame):
            names_col = mts.columns.values + "_" + str(i + 1)
        elif isinstance(mts, pd.Series):
            names_col = mts.name + "_" + str(i + 1)
        else:
            print('error!')
            exit(0)
        for j in range(len(names_col)):
            new_mts[names_col[j]] = i_data[mts.columns.values[j]].values
    return new_mts

class TestCrossCov:
    def __init__(self):
        1+1

    def fit(self, x, y):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        p_value = cross_corr(x, y)
        return 1-p_value[1]

class TsModel:
    def __init__(self, mts, method="var", score="aic", max_order=5, min_order=1):
        self.get_method = {
            "var": self.train_var,
            "gp": self.train_gp
            # "gam": self.train_gam
        }
        self.get_score = {
            "aic": aic,
            "bic": bic
        }
        self.mts = mts
        self.method = method
        self.score = score
        self.max_order = max_order
        self.min_order = min_order
        self.info = {
            "best_order": min_order,
            "score": 0
        }
        self.model = None
        self.X = None
        self.y = None

    def ts_order(self, ts, mts=None, order=4):
        new_mts = pd.DataFrame()
        for i in range(order):
            i_data = ts[i:(-order+i)].values
            new_mts.loc[:, ts.name+"_"+str(i+1)] = i_data

        if mts is not None:
            for i in range(order+1):
                if i == order:
                    i_data = mts[i:]
                else:
                    i_data = mts[i:(-order+i)]
                if isinstance(mts, pd.DataFrame):
                    names_col = mts.columns.values+"_"+str(i+1)
                elif isinstance(mts, pd.Series):
                    names_col = mts.name + "_" + str(i+1)
                else:
                    print('error!')
                    exit(0)
                for j in range(len(names_col)):
                    new_mts[names_col[j]] = i_data[mts.columns.values[j]].values

        ts = ts[order:]
        return ts, new_mts

    def train_var(self, name_y):
        order_list = list(range(self.min_order, self.max_order))
        if isinstance(self.mts, pd.DataFrame):
            names_x = list(self.mts.columns.values)
            names_x.remove(name_y)
            temp_x = self.mts[names_x]
            temp_y = self.mts[name_y]
        elif isinstance(self.mts, pd.Series):
            temp_x = None
            temp_y = self.mts
        else:
            print('error!!!')
            exit(0)
        scores = dict()
        for ord in order_list:
            y, X = self.ts_order(temp_y, temp_x, order=ord)
            model = Lr()
            model_fit = model.fit(X.values, y.values)
            pred = model_fit.predict(X.values)
            scores[ord] = self.get_score[self.score](y.values, pred, X.shape[1])

        best_order = min(scores, key=scores.get)
        self.info["best_order"] = best_order
        self.info["score"] = scores[best_order]

        self.y, self.X = self.ts_order(temp_y, temp_x, order=best_order)
        model = Lr()
        model_fit = model.fit(self.X.values, self.y.values)
        return model_fit

    # def train_gam(self, name_y):
        # model = LinearGAM(n_splines=10)
        # model = LinearGAM(s(0) + s(1) + f(2))
        # model.gridsearch(X, y)

        # return 0

    def train_gp(self):
        return 0

    def fit(self, name_y):
        self.model = self.get_method[self.method](name_y)

    def predict(self, test = None):
        if test == None:
            pred = self.model.predict(self.X.values[:-self.info["best_order"],:])
        else:
            pred = self.model.predict(test)
        return pred

    def residuals(self,test_x = None, test_y = None):
        if test_x == None:
            pred = self.model.predict(self.X.values)
            res = residuals(self.y.values, pred)
        else:
            pred = self.model.predict(test_x)
            res = residuals(test_y, pred)
        return res



from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i].reshape(-1,1), y[j].reshape(-1,1))
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

if __name__ == "__main__":
    import statsmodels.api as sm
    #
    mdata = sm.datasets.macrodata.load_pandas().data
    mdata = mdata[['realgdp', 'realcons', 'realinv']]
    # mdata = mdata[:200]
    # # print(mdata)
    # model1 = TsModel(mdata["realgdp"], max_order=6)
    # y, X = model1.ts_order(mdata["realgdp"], mdata[['realcons']].copy(), order=0)
    # print(X)
    print(mdata["realgdp"].shape)
    y = ts_order(mdata["realgdp"], order=2)
    print(y.shape)
    print(y.columns)
    #
    # model1.train_var("realgdp")
    #
    # tcv = TestCrossCov()
    # print(tcv.fit(mdata["realgdp"].values,mdata["realinv"].values))
    #
    # from tools.test_hsic import TestHSIC
    #
    # tcv = TestHSIC(method='gamma')
    # print(tcv.fit(mdata["realgdp"].values, mdata["realinv"].values))



    # a = np.zeros([4,4])
    # b = np.zeros([4,4])
    # a[0, 2] = 1
    # a[1, 2] = 1
    # a[2, 3] = 1
    # b[1, 2] = 1
    # b[2, 3] = 1
    # b[1, 3] = 1
    #
    # print(a)
    # print(b)
    # print(hamming_distance(a,b))