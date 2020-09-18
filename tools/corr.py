"""
sdfdsfdfdfd
Date: Feb 2019
Author: Karim Assaad, karimassaad3@gmail.com
"""

import numpy as np
from scipy import stats, linalg
import matplotlib.pyplot as plt

def partial_corr(C):
    """
    Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial
    correlation (might be slow for a huge number of variables). The
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
    the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from Steps #2 and #4;
        The result is the partial correlation between X and Y while controlling for the effect of Z
    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    github: https://gist.github.com/fabianp/9396204419c7b638d38f

    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def acf_one(x, y, lag=None):
    """
    sdsdsd
    :param x:
    :param y:
    :param lag:
    :return:
    """
    res = []
    for t in range(lag):
        c = np.corrcoef(np.array([x[0:len(x)-t], y[t:len(x)]]))[1,0]
        res.append(c)
    return res


def acf(x, y, lag=None, plot=True):
    """
    sdsds
    :param x:
    :param y:
    :param lag:
    :param plot:
    :return:
    """
    [N,] = x.shape
    if not lag:
        lag = int(round(10*np.log10(N/2)))
    res_xy = acf_one(x, y, lag=lag)
    res_yx = acf_one(y, x, lag=lag)
    res = np.concatenate((res_xy[::-1][:-1], res_yx))

    if plot:
        a = list(range(-lag+1,lag))
        plt.bar(a, res, width=0.4)
        plt.show()

    return res


# works only if eps is white noise!! (i.e. acf(k) = 0 for k!=0)
class TestCrossCov:
    def __init__(self):
        1

    def fit_uni(self, x, eps, alpha, max_lag):
        """
        test of cross correlation in Python (clone of R's TSindtest)
        :param x:
        :param eps:
        :param alpha:
        :param max_lag:
        :return:
        """
        corr1 = acf(x, eps, lag=max_lag, plot = False)
        T = max(abs(corr1))
        sigma = np.zeros([2*max_lag,2*max_lag])
        # Theorem 11.2.3 in brockwell and davis: "bartletts formula"
        # H_0: rho_{12} == 0 => non-zero summands only for j = k - h
        acr = acf_one(x, x, lag=2*max_lag)
        for i in range(2*max_lag):
            for j in range(i+1):
                sigma[i,j]=acr[(i-j)]
                sigma[j,i]=acr[(i-j)]
        sigma = sigma/len(x)
        R = np.linalg.cholesky(sigma)
        num_simulations = 20000
        z = np.random.randn(num_simulations,2*max_lag)
        z = np.dot(z, R)
        maxz = [max(abs(i)) for i in z]
        maxzorder = np.sort(maxz)
        quan = maxzorder[int(np.ceil(num_simulations-alpha*num_simulations))]
        p_value = sum(maxzorder>T)/num_simulations
        result = {"statistic": T, "crit_value": quan, "p_value": p_value}
        return result

    def fit(self, z, r1, alpha, max_lag, plotit=False):
        """
        test of cross correlation in Python (clone of R's TSindtest) for multivariate case
        :param z: numpy array
        :param r1:
        :param alpha:
        :param max_lag:
        :param plotit:
        :return:
        """
        # z = as.matrix(z)
        pdim2 = z.shape
        Tvecquanvec = []
        Tvec = np.zeros([pdim2[1], 1])
        quanvec = np.zeros([pdim2[1], 1])
        for i in range(pdim2[1]):
            # Tvecquanvec.append(self.fit_uni(z[names[i]], r1, alpha / pdim2[1], max_lag))
            Tvecquanvec.append(self.fit_uni(z[:,i], r1, alpha / pdim2[1], max_lag))
            Tvec[i] = Tvecquanvec[i]["statistic"]
            quanvec[i] = Tvecquanvec[i]["crit_value"]

        bb = np.argmax(Tvec-quanvec)
        T = float(Tvec[bb])
        quan = float(quanvec[bb])
        pval = Tvecquanvec[bb]["p_value"] * pdim2[1]
        resu = {"statistic": T, "crit_value": quan, "p_value": pval}
        return resu



if __name__ == "__main__":
    from data.sim_data import generate_v_structure
    mdata = generate_v_structure()

    x = mdata[['V1','V2']]
    y = mdata['V2']
    z = mdata['V3']


    # a = acf(x,y)
    # print(a)
    tcc = TestCrossCov()
    print(tcc.fit(x.values,y.values,0.05,5))

