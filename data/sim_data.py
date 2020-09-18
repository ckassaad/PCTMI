"""
Simulation of a V structure
Date: Jan 2019
Author: Karim Assaad, karimassaad3@gmail.com
"""

import pandas as pd
import numpy as np
import random


def func_add(x1, x2):
    return x1 + x2


def generate_v_structure(N=1000):
    print("V-Structure: 0 -> 2 <- 1")
    N = N + 1
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = 0.8 * y[0] + 0.5 * epsy[1]
    x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    y[2] = 0.8 * y[1] + 0.5 * epsy[2]
    w[2] = -0.6 * w[1] + 0.8 * x[0] + 0.8*y[1] + 0.5 * epsw[2]
    for i in range(3,N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.8*y[i-1] + 0.5*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data


def generate_fork_space(N=1000):
    N = N+3
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.1 * epsx[0]
    x[1] = 0.1 * epsx[1]
    x[2] = 0.1 * epsx[2]
    for i in range(3,N):
        x[i] = 0.3*x[i-1] + 0.3*x[i-2] + 0.3*x[i-3] +0.1*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1] +0.1*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2]  + 0.1*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.drop(data.index[0])
    data = data.drop(data.index[0])
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data


def generate_fork(N=1000):
    N = N+1
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = 0.8 * y[0] + 0.8 * x[0] + 0.5 * epsy[1]
    x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    y[2] = 0.8 * y[1] + 0.8 * x[1] + 0.5 * epsy[2]
    w[2] = -0.6 * w[1] + 0.8 * x[0] + 0.5 * epsw[2]
    for i in range(3,N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.1*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data


def generate_mediator(N=1000):
    print("Mediator: 0 -> 1 -> 2 <- 0")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    for i in range(3,N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.8*y[i-1] +0.5*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    return data


def generate_diamond(N=1000):
    print("Diamond: 3 <- 1 <- 0 -> 2 -> 3")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3
    epsz = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    for i in range(3, N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.5*epsw[i]
        z[i] = -0.6*z[i-1] + 0.8*y[i-2] + +0.8*w[i-1] + 0.5*epsz[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    data = pd.concat([x, y, w, z], axis=1, sort=False)
    return data


def generate_fork_nl(N=1000):
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.1 * epsx[0]
    x[1] = 0.3 * x[0] + 0.1 * epsx[1]
    y[1] = 0.8 * y[0] + 0.8 * x[0] + 0.5 * epsy[1]
    x[2] = 0.3 * x[1] + 0.1 * epsx[2]
    y[2] =  0.8*y[1]+0.8*(x[1])+0.5*epsy[2]
    w[2] = -0.6 * w[1] + 0.8 * (x[0]**2) + 0.1 * epsw[2]
    for i in range(3,N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*(x[i-1])+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*(x[i-2]**2) +0.1*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    return data


def generate_fork_nl_biglag(N=1000):
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = 0.8 * y[0] + 0.8 * x[0] + 0.5 * epsy[1]
    x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    y[2] = 0.8 * y[1] + 0.8 * x[1] + 0.5 * epsy[2]
    for i in range(3,9):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]**3+0.5*epsy[i]
    w[8] = -0.6 * w[7] + 0.8 * x[0] + 0.5 * epsw[7]
    for i in range(9,N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]**3+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-8]**2 +0.5*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    return data


def generate_v_structure_diff_sampling_rate(N=1000):
    N=N+1
    print("V-Structure: 0 -> 2 <- 1")
    epsw = np.random.randn(N) ** 3
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    w[0] = np.nan
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = np.nan
    x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    y[2] = 0.8 * y[0] + 0.5 * epsy[2]
    w[2] = np.nan
    for i in range(3, N):
        x[i] = 0.3 * x[i - 1] + 0.5 * epsx[i]
        if i % 2 == 0:
            y[i] = 0.8 * y[i - 2] + 0.5 * epsy[i]
            w[i] = np.nan
        else:
            y[i] = np.nan
            w[i] = -0.6*w[i-2]+0.8*x[i-2] + 0.8*y[i-1] +0.5*epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    return data

def generate_fork_diff_sampling_rate(N=1000):
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N) ** 3
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = 0.8 * y[0] + 0.8 * x[0] + 0.5 * epsy[1]
    w[1] = np.nan
    # x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    # y[2] = 0.8 * y[1] + 0.8 * x[1] + 0.5 * epsy[2]
    # w[2] = np.nan
    for i in range(2, N):
        x[i] = 0.3 * x[i - 1] + 0.5 * epsx[i]
        y[i] = 0.8 * y[i - 1] + 0.8 * x[i - 1] + 0.5 * epsy[i]
        if i % 2 == 0:
            w[i] = -0.6 * w[i - 2] + 0.8 * x[i - 2] + 0.5 * epsw[i]
        else:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    return data


def generate_diamond_diff_sampling_rate(N=1000):
    print("Diamond: 3 <- 1 <- 0 -> 2 -> 3")
    epsw = np.random.randn(N)**3
    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3
    epsz = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    for i in range(3, N):
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.8*x[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.5*epsw[i]
        z[i] = -0.6*z[i-1] + 0.8*y[i-2] + +0.8*w[i-1] + 0.5*epsz[i]

    j=0
    for i in range(N):
        if j == 0:
            y[i] = np.nan
            w[i] = np.nan
        if j == 1:
            w[i] = np.nan
            j = -2
        j = j+1

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    data = pd.concat([y, w, z], axis=1, sort=False)
    return data



def generate_v_structure_missing_data(N=1000):
    print("V-Structure: 0 -> 2 <- 1")
    import datetime
    dt = datetime.datetime(2010, 12, 1)
    # end = datetime.datetime(2010, 12, 30, 23, 59, 59)
    step = datetime.timedelta(hours=1)

    epsx = np.random.randn(N)**3
    epsy = np.random.randn(N)**3
    epsw = np.random.randn(N)**3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    d = list()

    for i in range(3):
        d.append(dt.strftime('%Y-%m-%d %H'))
        dt += step
    for i in range(3,N):
        d.append(dt.strftime('%Y-%m-%d %H'))
        dt += step
        x[i] = 0.3*x[i-1]+0.5*epsx[i]
        y[i] = 0.8*y[i-1]+0.5*epsy[i]
        w[i] = -0.6*w[i-1]+0.8*x[i-2] + 0.8*y[i-1] +0.5*epsw[i]

    for i in range(0,N):
        if (i+1)%2 != 0:
            y[i] = np.nan
        if (i+1)%3 != 0:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    data = data.set_index([pd.Index(d)])
    return data



def generate_fork_missing_data(N=1000):
    print("Fork: 1 <- 0 -> 2")
    epsw = np.random.randn(N) ** 3
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = 0.5 * epsx[0]
    x[1] = 0.3 * x[0] + 0.5 * epsx[1]
    y[1] = 0.8 * y[0] + 0.8 * x[0] + 0.5 * epsy[1]
    x[2] = 0.3 * x[1] + 0.5 * epsx[2]
    y[2] = 0.8 * y[1] + 0.8 * x[1] + 0.5 * epsy[2]
    w[2] = -0.6 * w[1] + 0.8 * x[0] + 0.5 * epsw[2]
    for i in range(3, N):
        x[i] = 0.3 * x[i - 1] + 0.5 * epsx[i]
        y[i] = 0.8 * y[i - 1] + 0.8 * x[i - 1] + 0.5 * epsy[i]
        w[i] = -0.6 * w[i - 1] + 0.8 * x[i - 2] + 0.5 * epsw[i]

    for i in range(0,N):
        if (i+1)%2 != 0:
            y[i] = np.nan
        if (i+1)%3 != 0:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    data = pd.concat([x, y, w], axis=1, sort=False)
    return data


def abso(x):
    return np.sqrt(np.power(x,2))

def identity(x):
    return x

functions_set = {0: abso, 1: np.tanh, 2:np.sin, 3: np.cos}


def uniform_with_gap(min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    while True:
        r = random.uniform(min_value, max_value)
        if min_gap>r or max_gap<r:
            break
    return r


def fork_generator_space(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    temporal = dict()
    temporal[0] = [(0, -1), (0, -2), (0, -3)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (2, -2), (0, -2)]

    N=N+2
    ax1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ax2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ax3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    aw2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity

    print("Fork: 1 <- 0 -> 2")
    print(f, g, ax1, ax2, ax3, bx, ay,by, aw1, aw2, bw, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = bx * epsx[1]
    w[1] = bw * epsw[1]
    x[2] = bx * epsx[2]
    y[2] = by * epsy[2]
    w[2] = aw1 * w[1] + aw2 * w[0] + axw * x[0] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax1 * x[i - 1] + ax2 * x[i - 2] + ax3 * x[i - 3] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        w[i] = aw1 * w[i - 1] + aw2 * w[i - 2] + axw * g(x[i - 2]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def indep_pair_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.zeros([2,2])))
    unit[0,1] = 0
    unit[1,0] = 0
    temporal = dict()
    temporal[0] = []
    temporal[1] = []

    N=N+2
    # ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    # ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1

    print("Independent Pair: 0 indep 1")
    print(bx, by)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])

    x[0] = bx * epsx[0]
    y[0] = by * epsy[0]
    for i in range(1, N):
        x[i] = bx * epsx[i]
        y[i] = by * epsy[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])

    series = pd.concat([x, y], axis=1, sort=False)
    series = series.drop(series.index[[0, 1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def pair_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([2,2])))
    unit[0,1] = 2
    unit[1,0] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity

    print("Pair: 0 -> 1")
    print(f, ax, bx, ay, by, axy)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])

    series = pd.concat([x, y], axis=1, sort=False)
    series = series.drop(series.index[[0, 1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal



def fork_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity

    print("Fork: 1 <- 0 -> 2")
    print(f, g, ax, bx, ay,by, aw, bw, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * g(x[i - 2]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def v_structure_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1, 2] = 2
    unit[2,1] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1)]
    temporal[2] = [(2, -1), (0, -2), (1, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity

    print("V-structure: 0 -> 2 <- 1")
    print(f, g, ax, bx, ay,by, aw, bw, axw, ayw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    y[0] = by * epsy[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + ayw * y[1] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * f(x[i - 2]) + ayw * g(y[i - 1]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def cycle_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3, 3])))
    unit[0, 1] = 2
    unit[1, 0] = 2
    unit[0, 2] = 2
    unit[2, 0] = 1
    temporal = dict()
    temporal[0] = [(0, -1), (0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayx = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set) - 1)]
    else:
        f = identity
        g = identity
        h = identity

    print("Cycle: 1 <-> 0 -> 2")
    print(f, g, h, ax, bx, ay,by, aw, bw, ayx, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + ayx * y[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + ayx * f(y[i - 1]) + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * g(x[i - 1]) + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * h(x[i - 2]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0, 1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal

# todo
def mediator_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3, 3])))
    unit[0, 1] = 2
    unit[1, 0] = 1
    unit[0, 2] = 2
    unit[2, 0] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1

    temporal = dict()
    temporal[0] = [(0, -1), (0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2), (1, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayx = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set) - 1)]
    else:
        f = identity
        g = identity
        h = identity

    print("Mediator: 1 <- 0 -> 2 <- 1")
    print(f, g, h, ax, bx, ay,by, aw, bw, ayx, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + ayx * y[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + ayw * y[1] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * g(x[i - 2]) + ayw * h(y[i - 1]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def diamond_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([4,4])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1, 3] = 2
    unit[3, 1] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]
    temporal[3] = [(3, -1), (1, -1), (2, -1)]

    N=N+3
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1
    az = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bz = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayz = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    awz = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set)-1)]
        k = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity
        h = identity
        k = identity

    print("Diamond: 3 <- 1 <- 0 -> 2 -> 3")
    print(f, g, h, k, ax, bx, ay,by, aw, bw, az, bz, axy, axw, ayz, awz)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3
    epsz = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + bw * epsw[2]
    x[3] = ax * x[2] + bx * epsx[3]
    y[3] = ay * y[2] + axy * x[2] + by * epsy[3]
    w[3] = aw * w[2] + axw * x[1] + bw * epsw[3]
    z[3] = az * z[2] + ayz * h(y[2]) + awz * k(w[2]) + bz * epsz[3]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * g(x[i - 2]) + bw * epsw[i]
        z[i] = az * z[i - 1] + ayz * h(y[i - 1]) + awz * k(w[i - 1]) + bz * epsz[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    series = pd.concat([x, y, w, z], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def v_structure_with_hidden_fork_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,1] = 2
    unit[1,0] = 2
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1
    temporal = dict()
    temporal[0] = [(0, -1), (1, -1)]
    temporal[1] = [(1, -1), (0, -2)]
    temporal[2] = [(2, -1), (0, -1), (1, -1)]

    N=N+3
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1
    az = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bz = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayz = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    awz = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set)-1)]
        k = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity
        h = identity
        k = identity

    print("Hidden: 2 <- 0 <-> 1 -> 2")
    print(f, g, h, k, ax, bx, ay,by, aw, bw, az, bz, axy, axw, ayz, awz)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3
    epsz = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[1] + axw * x[0] + bw * epsw[2]
    x[3] = ax * x[2] + bx * epsx[3]
    y[3] = ay * y[2] + axy * x[2] + by * epsy[3]
    w[3] = aw * w[2] + axw * x[1] + bw * epsw[3]
    z[3] = az * z[2] + ayz * h(y[2]) + awz * k(w[2]) + bz * epsz[3]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        w[i] = aw * w[i - 1] + axw * g(x[i - 2]) + bw * epsw[i]
        z[i] = az * z[i - 1] + ayz * h(y[i - 1]) + awz * k(w[i - 1]) + bz * epsz[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    series = pd.concat([x, y, w, z], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2]])
    series = series.reset_index(drop=True)
    series = series.drop(["V1"], axis=1)
    return series, unit, temporal

def structure_with_2_hidden_var_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.array([[1, 1, 0, 0, 0, 2, 0],
                      [2, 1, 1, 0, 0, 0, 2],
                      [0, 2, 1, 1, 0, 0, 0],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 1, 2, 0],
                      [2, 0, 0, 0, 1, 1, 2],
                      [0, 2, 0, 0, 0, 1, 1]])

    temporal = dict()
    temporal[0] = [(0, -1), (1, -1), (5, -1)]
    temporal[1] = [(1, -1), (0, -1), (2, -1)]
    temporal[2] = [(2, -1), (3, -1)]
    temporal[3] = [(3, -1)]
    temporal[4] = [(4, -1), (3, -1)]
    temporal[5] = [(5, -1), (4, -1), (6, -1)]
    temporal[6] = [(6, -1), (5, -1), (1, -1)]

    N=N+3
    at1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bt1 = 0.1
    at2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bt2 = 0.1
    aa = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ba = 0.1
    ab = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bb = 0.1
    ac = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bc = 0.1
    ad = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bd = 0.1
    ae = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    be = 0.1
    af = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bf = 0.1
    ah = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bh = 0.1

    atb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ata = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    abe = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    acf = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ach = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ada = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ate = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    atd = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    afb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ahd = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f0 = functions_set[random.randint(0, len(functions_set)-1)]
        f1 = functions_set[random.randint(0, len(functions_set)-1)]
        f2 = functions_set[random.randint(0, len(functions_set)-1)]
        f3 = functions_set[random.randint(0, len(functions_set)-1)]
        f4 = functions_set[random.randint(0, len(functions_set)-1)]
        f5 = functions_set[random.randint(0, len(functions_set)-1)]
        f6 = functions_set[random.randint(0, len(functions_set)-1)]
        f7 = functions_set[random.randint(0, len(functions_set)-1)]
        f8 = functions_set[random.randint(0, len(functions_set)-1)]
        f9 = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f0 = identity
        f1 = identity
        f2 = identity
        f3 = identity
        f4 = identity
        f5 = identity
        f6 = identity
        f7 = identity
        f8 = identity
        f9 = identity

    print("complex structure with two hidden causes")
    # print(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, atb, ata, abe, acf, ach, ada, ate, atd, afb, ahd)
    print(f8, f6, f2, f4, f9, f7, f0, f1, f3, f5, at1, bt1, at2, bt2, ae, be, ab, bb, af, bf, ac, bc, ah, bh, ad, bd,
          aa, ba, ate, atb, atd, ate, abe, afb, acf, ach, ahd, ada)

    epst1 = np.random.randn(N) ** 3
    epst2 = np.random.randn(N) ** 3
    epsa = np.random.randn(N) ** 3
    epsb = np.random.randn(N) ** 3
    epsc = np.random.randn(N) ** 3
    epsd = np.random.randn(N) ** 3
    epse = np.random.randn(N) ** 3
    epsf = np.random.randn(N) ** 3
    epsh = np.random.randn(N) ** 3

    t1 = np.zeros([N])
    t2 = np.zeros([N])
    a = np.zeros([N])
    b = np.zeros([N])
    c = np.zeros([N])
    d = np.zeros([N])
    e = np.zeros([N])
    f = np.zeros([N])
    h = np.zeros([N])

    t1[0] = bt1 * epst1[0]
    t2[0] = bt2 * epst2[0]
    c[0] = bc * epsc[0]
    f[0] = bf * epsf[0]
    h[0] = bh * epsh[0]
    d[0] = bd * epsd[0]
    a[0] = ba * epsa[0]
    b[0] = bb * epsb[0]
    e[0] = be * epse[0]

    t1[1] = at1 * t1[0] + bt1 * epst1[1]
    t2[1] = at2 * t2[0] + bt2 * epst2[1]
    c[1] = ac * c[0] + bc * epsc[1]
    f[1] = af * f[0] + acf * f0(c[0]) + bf * epsf[1]
    h[1] = ah * h[0] + ach * f1(c[0]) + bh * epsh[1]
    d[1] = ad * d[0] + atd * f2(t2[0]) + ahd * f3(h[0]) + bd * epsd[1]
    a[1] = aa * a[0] + ata * f4(t1[0]) + ada * f5(d[0]) + ba * epsa[1]
    b[1] = ab * b[0] + atb * f6(t1[0]) + afb * f7(f[0]) + bb * epsb[1]
    e[1] = ae * e[0] + ate * f8(t2[0]) + abe * f9(b[0]) + be * epse[1]

    for i in range(2, N):
        t1[i] = at1 * t1[i - 1] + bt1 * epst1[i]
        t2[i] = at2 * t2[i - 1] + bt2 * epst2[i]
        c[i] = ac * c[i - 1] + bc * epsc[i]

        f[i] = af * f[i - 1] + acf * f0(c[i - 1]) + bf * epsf[i]
        h[i] = ah * h[i - 1] + ach * f1(c[i - 1]) + bh * epsh[i]
        d[i] = ad * d[i - 1] + atd * f2(t2[i - 1]) + ahd * f3(h[i - 1]) + bd * epsd[i]

        a[i] = aa * a[i - 1] + ata * f4(t1[i - 1]) + ada * f5(d[i - 1]) + ba * epsa[i]
        b[i] = ab * b[i - 1] + atb * f6(t1[i - 1]) + afb * f7(f[i - 1]) + bb * epsb[i]

        e[i] = ae * e[i - 1] + ate * f8(t2[i - 1]) + abe * f9(b[i - 1]) + be * epse[i]

    a = pd.DataFrame(a, columns=["A"])
    b = pd.DataFrame(b, columns=["B"])
    f = pd.DataFrame(f, columns=["F"])
    c = pd.DataFrame(c, columns=["C"])
    h = pd.DataFrame(h, columns=["H"])
    d = pd.DataFrame(d, columns=["D"])
    e = pd.DataFrame(e, columns=["E"])

    series = pd.concat([e, b, f, c, h, d, a], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def seven_ts_generator(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.array([[1, 1, 0, 0, 0, 0, 0],
                      [2, 1, 1, 0, 0, 0, 0],
                      [0, 2, 1, 1, 0, 0, 0],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 1, 2, 0],
                      [0, 0, 0, 0, 1, 1, 2],
                      [0, 0, 0, 0, 0, 1, 1]])

    temporal = dict()
    temporal[0] = [(0, -1), (1, -1), (5, -1)]
    temporal[1] = [(1, -1), (0, -1), (2, -1)]
    temporal[2] = [(2, -1), (3, -1)]
    temporal[3] = [(3, -1)]
    temporal[4] = [(4, -1), (3, -1)]
    temporal[5] = [(5, -1), (4, -1), (6, -1)]
    temporal[6] = [(6, -1), (5, -1), (1, -1)]

    N=N+3
    # at1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # bt1 = 0.1
    # at2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # bt2 = 0.1
    aa = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ba = 0.1
    ab = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bb = 0.1
    ac = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bc = 0.1
    ad = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bd = 0.1
    ae = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    be = 0.1
    af = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bf = 0.1
    ah = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bh = 0.1

    # atb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # ata = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    abe = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    acf = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ach = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ada = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # ate = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # atd = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    afb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ahd = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f0 = functions_set[random.randint(0, len(functions_set)-1)]
        f1 = functions_set[random.randint(0, len(functions_set)-1)]
        # f2 = functions_set[random.randint(0, len(functions_set)-1)]
        f3 = functions_set[random.randint(0, len(functions_set)-1)]
        # f4 = functions_set[random.randint(0, len(functions_set)-1)]
        f5 = functions_set[random.randint(0, len(functions_set)-1)]
        # f6 = functions_set[random.randint(0, len(functions_set)-1)]
        f7 = functions_set[random.randint(0, len(functions_set)-1)]
        # f8 = functions_set[random.randint(0, len(functions_set)-1)]
        f9 = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f0 = identity
        f1 = identity
        # f2 = identity
        f3 = identity
        # f4 = identity
        f5 = identity
        # f6 = identity
        f7 = identity
        # f8 = identity
        f9 = identity

    print("7 time series structure with no hidden causes")
    print(f9, f7, f0, f1, f3, f5, ae, be, ab, bb, af, bf, ac, bc, ah, bh, ad, bd, aa, ba, abe, afb, acf, ach, ahd, ada)
    # epst1 = np.random.randn(N) ** 3
    # epst2 = np.random.randn(N) ** 3
    epsa = np.random.randn(N) ** 3
    epsb = np.random.randn(N) ** 3
    epsc = np.random.randn(N) ** 3
    epsd = np.random.randn(N) ** 3
    epse = np.random.randn(N) ** 3
    epsf = np.random.randn(N) ** 3
    epsh = np.random.randn(N) ** 3

    # t1 = np.zeros([N])
    # t2 = np.zeros([N])
    a = np.zeros([N])
    b = np.zeros([N])
    c = np.zeros([N])
    d = np.zeros([N])
    e = np.zeros([N])
    f = np.zeros([N])
    h = np.zeros([N])

    # t1[0] = bt1 * epst1[0]
    # t2[0] = bt2 * epst2[0]
    c[0] = bc * epsc[0]
    f[0] = bf * epsf[0]
    h[0] = bh * epsh[0]
    d[0] = bd * epsd[0]
    a[0] = ba * epsa[0]
    b[0] = bb * epsb[0]
    e[0] = be * epse[0]

    # t1[1] = at1 * t1[0] + bt1 * epst1[1]
    # t2[1] = at2 * t2[0] + bt2 * epst2[1]
    c[1] = ac * c[0] + bc * epsc[1]
    f[1] = af * f[0] + acf * f0(c[0]) + bf * epsf[1]
    h[1] = ah * h[0] + ach * f1(c[0]) + bh * epsh[1]
    d[1] = ad * d[0] + ahd * f3(h[0]) + bd * epsd[1]
    a[1] = aa * a[0] + ada * f5(d[0]) + ba * epsa[1]
    b[1] = ab * b[0] + afb * f7(f[0]) + bb * epsb[1]
    e[1] = ae * e[0] + abe * f9(b[0]) + be * epse[1]

    for i in range(2, N):
        # t1[i] = at1 * t1[i - 1] + bt1 * epst1[i]
        # t2[i] = at2 * t2[i - 1] + bt2 * epst2[i]
        c[i] = ac * c[i - 1] + bc * epsc[i]

        f[i] = af * f[i - 1] + acf * f0(c[i - 1]) + bf * epsf[i]
        h[i] = ah * h[i - 1] + ach * f1(c[i - 1]) + bh * epsh[i]
        d[i] = ad * d[i - 1] + ahd * f3(h[i - 1]) + bd * epsd[i]

        a[i] = aa * a[i - 1] + ada * f5(d[i - 1]) + ba * epsa[i]
        b[i] = ab * b[i - 1] + afb * f7(f[i - 1]) + bb * epsb[i]

        e[i] = ae * e[i - 1] + abe * f9(b[i - 1]) + be * epse[i]

    a = pd.DataFrame(a, columns=["A"])
    b = pd.DataFrame(b, columns=["B"])
    f = pd.DataFrame(f, columns=["F"])
    c = pd.DataFrame(c, columns=["C"])
    h = pd.DataFrame(h, columns=["H"])
    d = pd.DataFrame(d, columns=["D"])
    e = pd.DataFrame(e, columns=["E"])

    series = pd.concat([e, b, f, c, h, d, a], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def mooij_7ts(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([7,7])
    unit = np.diag(np.diag(np.ones([7,7])))

    unit[2, 1] = 2
    unit[1, 2] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1
    unit[2, 6] = 2
    unit[6, 2] = 1
    unit[1, 3] = 2
    unit[3, 1] = 1
    unit[1, 5] = 2
    unit[5, 1] = 1
    unit[1, 6] = 2
    unit[6, 1] = 1
    unit[1, 0] = 2
    unit[0, 1] = 1
    unit[3, 5] = 2
    unit[5, 3] = 1
    unit[3, 0] = 2
    unit[0, 3] = 1
    unit[5, 6] = 2
    unit[6, 5] = 1
    unit[5, 4] = 2
    unit[4, 5] = 1
    unit[6, 0] = 2
    unit[0, 6] = 1
    unit[6, 4] = 2
    unit[4, 6] = 1

    temporal = dict()
    temporal[2] = [(2, -1)]
    temporal[1] = [(1, -1), (2, -1)]
    temporal[3] = [(3, -1), (2, -1), (1, -1)]
    temporal[5] = [(5, -1), (1, -1), (3, -1)]
    temporal[6] = [(6, -1), (2, -1), (5, -1)]
    temporal[0] = [(0, -1), (1, -1), (3, -1), (6, -1)]
    temporal[4] = [(4, -1), (1, -1), (5, -1), (6, -1)]

    N=N+5
    a1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b1 = 0.1
    a2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b2 = 0.1
    a3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b3 = 0.1
    a4 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b4 = 0.1
    a5 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b5 = 0.1
    a6 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b6 = 0.1
    a7 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b7 = 0.1

    print("Complex structure")
    print(a1, a2, a3, a4, a5, a6, a7)
    eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    eps2 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps3 = np.random.uniform(low=-1.0, high=1.0, size=N)
    eps4 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps5 = np.random.uniform(low=-0.2, high=0.2, size=N)
    eps6 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps7 = np.random.uniform(low=-0.3, high=0.3, size=N)

    # eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps2 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps3 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps4 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps5 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps6 = np.random.uniform(low=-0.1, high=0.1, size=N)
    # eps7 = np.random.uniform(low=-0.1, high=0.1, size=N)

    # eps1 = np.random.randn(N) ** 3
    # eps2 = np.random.randn(N) ** 3
    # eps3 = np.random.randn(N) ** 3
    # eps4 = np.random.randn(N) ** 3
    # eps5 = np.random.randn(N) ** 3
    # eps6 = np.random.randn(N) ** 3
    # eps7 = np.random.randn(N) ** 3

    v1 = np.zeros([N])
    v2 = np.zeros([N])
    v3 = np.zeros([N])
    v4 = np.zeros([N])
    v5 = np.zeros([N])
    v6 = np.zeros([N])
    v7 = np.zeros([N])

    v3[0] = b3 * eps3[0]
    v3[1] = a3 * v3[0] + b3 * eps3[1]
    v2[1] = a2 * v2[0] + np.power(v3[0], 2) + b2 * eps2[1]
    v3[2] = a3 * v3[1] + b3 * eps3[2]
    v2[2] = a2 * v2[1] + np.power(v3[1], 2) + b2 * eps2[2]
    v4[2] = a4 * v4[1] + np.sin(v2[1]) + np.sin(2 * v3[1]) + b4 * eps4[2]
    v3[3] = a3 * v3[2] + b3 * eps3[3]
    v2[3] = a2 * v2[2] + np.power(v3[2], 2) + b2 * eps2[3]
    v4[3] = a4 * v4[2] + np.sin(v2[2]) + np.sin(2 * v3[2]) + b4 * eps4[3]
    v6[3] = a6 * v6[2] + np.sin(v2[2]) + np.cos(2 * v4[2]) + b6 * eps6[3]
    v3[4] = a3 * v3[3] + b3 * eps3[4]
    v2[4] = a2 * v2[3] + np.power(v3[3], 3) + b2 * eps2[4]
    v4[4] = a4 * v4[3] + np.sin(v2[3]) + np.sin(2 * v3[3]) + b4 * eps4[4]
    v6[4] = a6 * v6[3] + np.sin(v2[3]) + np.cos(2 * v4[3]) + b6 * eps6[4]
    v7[4] = a7 * v7[3] + np.cos(v6[3] + v3[3]) + b7 * eps7[4]
    v3[5] = a3 * v3[4] + b3 * eps3[5]
    v2[5] = a2 * v2[4] + np.power(v3[4], 3) + b2 * eps2[5]
    v4[5] = a4 * v4[4] + np.sin(v2[4]) + np.sin(2 * v3[4]) + b4 * eps4[5]
    v6[5] = a6 * v6[4] + np.sin(v2[4]) + np.cos(2 * v4[4]) + b6 * eps6[5]
    v7[5] = a7 * v7[4] + np.cos(v6[4] + v3[4]) + b7 * eps7[5]
    v1[5] = a1 * v1[4] + np.sin(np.power(v4[4], 2)) + np.power(v2[4], 2) + np.cos(v7[4]) + b1 * eps1[5]
    v5[5] = a5 * v5[4] + np.tanh(v6[4] + v7[4] + v2[4]) + b5 * eps5[5]
    for i in range(6, N):
        v3[i] = a3 * v3[i - 1] + b3 * eps3[i]
        v2[i] = a2 * v2[i - 1] + np.power(v3[i-1], 2) + b2 * eps2[i]
        v4[i] = a4 * v4[i - 1] + np.sin(v2[i-1]) + np.sin(2 * v3[i-1]) + b4 * eps4[i]
        v6[i] = a6 * v6[i - 1] + np.sin(v2[i-1]) + np.cos(2 * v4[i-1]) + b6 * eps6[i]
        v7[i] = a7 * v7[i - 1] + np.cos(v6[i-1] + v3[i-1]) + b7 * eps7[i]
        v1[i] = a1 * v1[i - 1] + np.sin(np.power(v4[i-1], 2)) + np.power(v2[i-1], 2) + np.cos(v7[i-1]) + b1 * eps1[i]
        v5[i] = a5 * v5[i - 1] + np.tanh(v6[i-1] + v7[i-1] + v2[i-1]) + b5 * eps5[i]

    v3 = pd.DataFrame(v3, columns=["V3"])
    v2 = pd.DataFrame(v2, columns=["V2"])
    v4 = pd.DataFrame(v4, columns=["V4"])
    v6 = pd.DataFrame(v6, columns=["V6"])
    v7 = pd.DataFrame(v7, columns=["V7"])
    v1 = pd.DataFrame(v1, columns=["V1"])
    v5 = pd.DataFrame(v5, columns=["V5"])

    series = pd.concat([v1, v2, v3, v4, v5, v6, v7], axis=1, sort=False)
    # series = pd.concat([v2, v3, v4, v6], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2, 3, 4]])
    series = series.reset_index(drop=True)
    return series, unit, temporal









def mooij_7ts_reduced(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([7,7])
    unit = np.diag(np.diag(np.ones([4,4])))

    unit[1, 0] = 2
    unit[0, 1] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1
    unit[0, 2] = 2
    unit[2, 0] = 1
    unit[0, 3] = 2
    unit[3, 0] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1

    temporal = dict()
    temporal[1] = [(1, -1)]
    temporal[0] = [(0, -1), (1, -1)]
    temporal[2] = [(2, -1), (1, -1), (0, -1)]
    temporal[3] = [(3, -1), (0, -1), (2, -1)]

    N=N+5
    a1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b1 = 0.1
    a2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b2 = 0.1
    a3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b3 = 0.1
    a4 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b4 = 0.1
    a5 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b5 = 0.1
    a6 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b6 = 0.1
    a7 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b7 = 0.1

    print("Complex structure")
    print(a1, a2, a3, a4, a5, a6, a7)
    eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    eps2 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps3 = np.random.uniform(low=-1.0, high=1.0, size=N)
    eps4 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps5 = np.random.uniform(low=-0.2, high=0.2, size=N)
    eps6 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps7 = np.random.uniform(low=-0.3, high=0.3, size=N)

    v1 = np.zeros([N])
    v2 = np.zeros([N])
    v3 = np.zeros([N])
    v4 = np.zeros([N])
    v5 = np.zeros([N])
    v6 = np.zeros([N])
    v7 = np.zeros([N])

    v3[0] = b3 * eps3[0]
    v3[1] = a3 * v3[0] + b3 * eps3[1]
    v2[1] = a2 * v2[0] + np.power(v3[0], 2) + b2 * eps2[1]
    v3[2] = a3 * v3[1] + b3 * eps3[2]
    v2[2] = a2 * v2[1] + np.power(v3[1], 2) + b2 * eps2[2]
    v4[2] = a4 * v4[1] + np.sin(v2[1]) + np.sin(2 * v3[1]) + b4 * eps4[2]
    v3[3] = a3 * v3[2] + b3 * eps3[3]
    v2[3] = a2 * v2[2] + np.power(v3[2], 2) + b2 * eps2[3]
    v4[3] = a4 * v4[2] + np.sin(v2[2]) + np.sin(2 * v3[2]) + b4 * eps4[3]
    v6[3] = a6 * v6[2] + np.sin(v2[2]) + np.cos(2 * v4[2]) + b6 * eps6[3]
    v3[4] = a3 * v3[3] + b3 * eps3[4]
    v2[4] = a2 * v2[3] + np.power(v3[3], 3) + b2 * eps2[4]
    v4[4] = a4 * v4[3] + np.sin(v2[3]) + np.sin(2 * v3[3]) + b4 * eps4[4]
    v6[4] = a6 * v6[3] + np.sin(v2[3]) + np.cos(2 * v4[3]) + b6 * eps6[4]
    v7[4] = a7 * v7[3] + np.cos(v6[3] + v3[3]) + b7 * eps7[4]
    v3[5] = a3 * v3[4] + b3 * eps3[5]
    v2[5] = a2 * v2[4] + np.power(v3[4], 3) + b2 * eps2[5]
    v4[5] = a4 * v4[4] + np.sin(v2[4]) + np.sin(2 * v3[4]) + b4 * eps4[5]
    v6[5] = a6 * v6[4] + np.sin(v2[4]) + np.cos(2 * v4[4]) + b6 * eps6[5]
    v7[5] = a7 * v7[4] + np.cos(v6[4] + v3[4]) + b7 * eps7[5]
    v1[5] = a1 * v1[4] + np.sin(np.power(v4[4], 2)) + np.power(v2[4], 2) + np.cos(v7[4]) + b1 * eps1[5]
    v5[5] = a5 * v5[4] + np.tanh(v6[4] + v7[4] + v2[4]) + b5 * eps5[5]
    for i in range(6, N):
        v3[i] = a3 * v3[i - 1] + b3 * eps3[i]
        v2[i] = a2 * v2[i - 1] + np.power(v3[i-1], 2) + b2 * eps2[i]
        v4[i] = a4 * v4[i - 1] + np.sin(v2[i-1]) + np.sin(2 * v3[i-1]) + b4 * eps4[i]
        v6[i] = a6 * v6[i - 1] + np.sin(v2[i-1]) + np.cos(2 * v4[i-1]) + b6 * eps6[i]
        v7[i] = a7 * v7[i - 1] + np.cos(v6[i-1] + v3[i-1]) + b7 * eps7[i]
        v1[i] = a1 * v1[i - 1] + np.sin(np.power(v4[i-1], 2)) + np.power(v2[i-1], 2) + np.cos(v7[i-1]) + b1 * eps1[i]
        v5[i] = a5 * v5[i - 1] + np.tanh(v6[i-1] + v7[i-1] + v2[i-1]) + b5 * eps5[i]

    v3 = pd.DataFrame(v3, columns=["V3"])
    v2 = pd.DataFrame(v2, columns=["V2"])
    v4 = pd.DataFrame(v4, columns=["V4"])
    v6 = pd.DataFrame(v6, columns=["V6"])
    v7 = pd.DataFrame(v7, columns=["V7"])
    v1 = pd.DataFrame(v1, columns=["V1"])
    v5 = pd.DataFrame(v5, columns=["V5"])

    series = pd.concat([v2, v3, v4, v6], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2, 3, 4]])
    series = series.reset_index(drop=True)
    return series, unit, temporal




def mooij_7ts_reduced2(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([7,7])
    unit = np.diag(np.diag(np.ones([5,5])))

    unit[1, 0] = 2
    unit[0, 1] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1
    unit[0, 2] = 2
    unit[2, 0] = 1
    unit[0, 3] = 2
    unit[3, 0] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1

    unit[1, 4] = 2
    unit[4, 1] = 1
    unit[3, 4] = 2
    unit[4, 3] = 1

    temporal = dict()
    temporal[1] = [(1, -1)]
    temporal[0] = [(0, -1), (1, -1)]
    temporal[2] = [(2, -1), (1, -1), (0, -1)]
    temporal[3] = [(3, -1), (0, -1), (2, -1)]
    temporal[4] = [(4, -1), (1, -1), (3, -1)]

    N=N+5
    a1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b1 = 0.1
    a2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b2 = 0.1
    a3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b3 = 0.1
    a4 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b4 = 0.1
    a5 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b5 = 0.1
    a6 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b6 = 0.1
    a7 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b7 = 0.1

    print("Complex structure")
    print(a1, a2, a3, a4, a5, a6, a7)
    eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    eps2 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps3 = np.random.uniform(low=-1.0, high=1.0, size=N)
    eps4 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps5 = np.random.uniform(low=-0.2, high=0.2, size=N)
    eps6 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps7 = np.random.uniform(low=-0.3, high=0.3, size=N)

    v1 = np.zeros([N])
    v2 = np.zeros([N])
    v3 = np.zeros([N])
    v4 = np.zeros([N])
    v5 = np.zeros([N])
    v6 = np.zeros([N])
    v7 = np.zeros([N])

    v3[0] = b3 * eps3[0]
    v3[1] = a3 * v3[0] + b3 * eps3[1]
    v2[1] = a2 * v2[0] + np.power(v3[0], 2) + b2 * eps2[1]
    v3[2] = a3 * v3[1] + b3 * eps3[2]
    v2[2] = a2 * v2[1] + np.power(v3[1], 2) + b2 * eps2[2]
    v4[2] = a4 * v4[1] + np.sin(v2[1]) + np.sin(2 * v3[1]) + b4 * eps4[2]
    v3[3] = a3 * v3[2] + b3 * eps3[3]
    v2[3] = a2 * v2[2] + np.power(v3[2], 2) + b2 * eps2[3]
    v4[3] = a4 * v4[2] + np.sin(v2[2]) + np.sin(2 * v3[2]) + b4 * eps4[3]
    v6[3] = a6 * v6[2] + np.sin(v2[2]) + np.cos(2 * v4[2]) + b6 * eps6[3]
    v3[4] = a3 * v3[3] + b3 * eps3[4]
    v2[4] = a2 * v2[3] + np.power(v3[3], 3) + b2 * eps2[4]
    v4[4] = a4 * v4[3] + np.sin(v2[3]) + np.sin(2 * v3[3]) + b4 * eps4[4]
    v6[4] = a6 * v6[3] + np.sin(v2[3]) + np.cos(2 * v4[3]) + b6 * eps6[4]
    v7[4] = a7 * v7[3] + np.cos(v6[3] + v3[3]) + b7 * eps7[4]
    v3[5] = a3 * v3[4] + b3 * eps3[5]
    v2[5] = a2 * v2[4] + np.power(v3[4], 3) + b2 * eps2[5]
    v4[5] = a4 * v4[4] + np.sin(v2[4]) + np.sin(2 * v3[4]) + b4 * eps4[5]
    v6[5] = a6 * v6[4] + np.sin(v2[4]) + np.cos(2 * v4[4]) + b6 * eps6[5]
    v7[5] = a7 * v7[4] + np.cos(v6[4] + v3[4]) + b7 * eps7[5]
    v1[5] = a1 * v1[4] + np.sin(np.power(v4[4], 2)) + np.power(v2[4], 2) + np.cos(v7[4]) + b1 * eps1[5]
    v5[5] = a5 * v5[4] + np.tanh(v6[4] + v7[4] + v2[4]) + b5 * eps5[5]
    for i in range(6, N):
        v3[i] = a3 * v3[i - 1] + b3 * eps3[i]
        v2[i] = a2 * v2[i - 1] + np.power(v3[i-1], 2) + b2 * eps2[i]
        v4[i] = a4 * v4[i - 1] + np.sin(v2[i-1]) + np.sin(2 * v3[i-1]) + b4 * eps4[i]
        v6[i] = a6 * v6[i - 1] + np.sin(v2[i-1]) + np.cos(2 * v4[i-1]) + b6 * eps6[i]
        v7[i] = a7 * v7[i - 1] + np.cos(v6[i-1] + v3[i-1]) + b7 * eps7[i]
        v1[i] = a1 * v1[i - 1] + np.sin(np.power(v4[i-1], 2)) + np.power(v2[i-1], 2) + np.cos(v7[i-1]) + b1 * eps1[i]
        v5[i] = a5 * v5[i - 1] + np.tanh(v6[i-1] + v7[i-1] + v2[i-1]) + b5 * eps5[i]

    v3 = pd.DataFrame(v3, columns=["V3"])
    v2 = pd.DataFrame(v2, columns=["V2"])
    v4 = pd.DataFrame(v4, columns=["V4"])
    v6 = pd.DataFrame(v6, columns=["V6"])
    v7 = pd.DataFrame(v7, columns=["V7"])
    v1 = pd.DataFrame(v1, columns=["V1"])
    v5 = pd.DataFrame(v5, columns=["V5"])

    series = pd.concat([v2, v3, v4, v6, v7], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2, 3, 4]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def mooij_7ts_reduced3(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([7,7])
    unit = np.diag(np.diag(np.ones([6,6])))

    unit[1, 0] = 2
    unit[0, 1] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1
    unit[0, 2] = 2
    unit[2, 0] = 1

    unit[0, 4] = 2
    unit[4, 0] = 1
    unit[2, 4] = 2
    unit[4, 2] = 1

    unit[1, 5] = 2
    unit[5, 1] = 1
    unit[4, 5] = 2
    unit[5, 4] = 1

    unit[0, 3] = 2
    unit[3, 0] = 1
    unit[4, 3] = 2
    unit[3, 4] = 1
    unit[5, 3] = 2
    unit[3, 5] = 1


    temporal = dict()
    temporal[1] = [(1, -1)]
    temporal[0] = [(0, -1), (1, -1)]
    temporal[2] = [(2, -1), (1, -1), (0, -1)]
    temporal[3] = [(3, -1), (0, -1), (2, -1)]
    temporal[4] = [(4, -1), (1, -1), (3, -1)]

    N=N+5
    a1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b1 = 0.1
    a2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b2 = 0.1
    a3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b3 = 0.1
    a4 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b4 = 0.1
    a5 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b5 = 0.1
    a6 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b6 = 0.1
    a7 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b7 = 0.1

    print("Complex structure")
    print(a1, a2, a3, a4, a5, a6, a7)
    eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    eps2 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps3 = np.random.uniform(low=-1.0, high=1.0, size=N)
    eps4 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps5 = np.random.uniform(low=-0.2, high=0.2, size=N)
    eps6 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps7 = np.random.uniform(low=-0.3, high=0.3, size=N)

    v1 = np.zeros([N])
    v2 = np.zeros([N])
    v3 = np.zeros([N])
    v4 = np.zeros([N])
    v5 = np.zeros([N])
    v6 = np.zeros([N])
    v7 = np.zeros([N])

    v3[0] = b3 * eps3[0]
    v3[1] = a3 * v3[0] + b3 * eps3[1]
    v2[1] = a2 * v2[0] + np.power(v3[0], 2) + b2 * eps2[1]
    v3[2] = a3 * v3[1] + b3 * eps3[2]
    v2[2] = a2 * v2[1] + np.power(v3[1], 2) + b2 * eps2[2]
    v4[2] = a4 * v4[1] + np.sin(v2[1]) + np.sin(2 * v3[1]) + b4 * eps4[2]
    v3[3] = a3 * v3[2] + b3 * eps3[3]
    v2[3] = a2 * v2[2] + np.power(v3[2], 2) + b2 * eps2[3]
    v4[3] = a4 * v4[2] + np.sin(v2[2]) + np.sin(2 * v3[2]) + b4 * eps4[3]
    v6[3] = a6 * v6[2] + np.sin(v2[2]) + np.cos(2 * v4[2]) + b6 * eps6[3]
    v3[4] = a3 * v3[3] + b3 * eps3[4]
    v2[4] = a2 * v2[3] + np.power(v3[3], 3) + b2 * eps2[4]
    v4[4] = a4 * v4[3] + np.sin(v2[3]) + np.sin(2 * v3[3]) + b4 * eps4[4]
    v6[4] = a6 * v6[3] + np.sin(v2[3]) + np.cos(2 * v4[3]) + b6 * eps6[4]
    v7[4] = a7 * v7[3] + np.cos(v6[3] + v3[3]) + b7 * eps7[4]
    v3[5] = a3 * v3[4] + b3 * eps3[5]
    v2[5] = a2 * v2[4] + np.power(v3[4], 3) + b2 * eps2[5]
    v4[5] = a4 * v4[4] + np.sin(v2[4]) + np.sin(2 * v3[4]) + b4 * eps4[5]
    v6[5] = a6 * v6[4] + np.sin(v2[4]) + np.cos(2 * v4[4]) + b6 * eps6[5]
    v7[5] = a7 * v7[4] + np.cos(v6[4] + v3[4]) + b7 * eps7[5]
    v1[5] = a1 * v1[4] + np.sin(np.power(v4[4], 2)) + np.power(v2[4], 2) + np.cos(v7[4]) + b1 * eps1[5]
    v5[5] = a5 * v5[4] + np.tanh(v6[4] + v7[4] + v2[4]) + b5 * eps5[5]
    for i in range(6, N):
        v3[i] = a3 * v3[i - 1] + b3 * eps3[i]
        v2[i] = a2 * v2[i - 1] + np.power(v3[i-1], 2) + b2 * eps2[i]
        v4[i] = a4 * v4[i - 1] + np.sin(v2[i-1]) + np.sin(2 * v3[i-1]) + b4 * eps4[i]
        v6[i] = a6 * v6[i - 1] + np.sin(v2[i-1]) + np.cos(2 * v4[i-1]) + b6 * eps6[i]
        v7[i] = a7 * v7[i - 1] + np.cos(v6[i-1] + v3[i-1]) + b7 * eps7[i]
        v1[i] = a1 * v1[i - 1] + np.sin(np.power(v4[i-1], 2)) + np.power(v2[i-1], 2) + np.cos(v7[i-1]) + b1 * eps1[i]
        v5[i] = a5 * v5[i - 1] + np.tanh(v6[i-1] + v7[i-1] + v2[i-1]) + b5 * eps5[i]

    v3 = pd.DataFrame(v3, columns=["V3"])
    v2 = pd.DataFrame(v2, columns=["V2"])
    v4 = pd.DataFrame(v4, columns=["V4"])
    v6 = pd.DataFrame(v6, columns=["V6"])
    v7 = pd.DataFrame(v7, columns=["V7"])
    v1 = pd.DataFrame(v1, columns=["V1"])
    v5 = pd.DataFrame(v5, columns=["V5"])

    series = pd.concat([v2, v3, v4, v5, v6, v7], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2, 3, 4]])
    series = series.reset_index(drop=True)
    return series, unit, temporal

#########################################################################################

def pair_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([2,2])))
    unit[0,1] = 2
    unit[1,0] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity

    print("Pair: 0 -> 1")
    print(f, ax, bx, ay, by, axy)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = np.nan
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[0] + axy * x[1] + by * epsy[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        if i % 2 == 0:
            y[i] = ay * y[i - 2] + axy * f(x[i - 1]) + by * epsy[i]
        else:
            y[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])

    series = pd.concat([x, y], axis=1, sort=False)
    series = series.drop(series.index[[0, 1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def fork_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity

    print("Fork: 1 <- 0 -> 2")
    print(f, g, ax, bx, ay,by, aw, bw, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    w[1] = np.nan

    for i in range(2, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        if i % 2 == 0:
            w[i] = aw * w[i - 2] + axw * g(x[i - 2]) + bw * epsw[i]
        else:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def v_structure_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1, 2] = 2
    unit[2,1] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1)]
    temporal[2] = [(2, -1), (0, -2), (1, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity

    print("V-structure: 0 -> 2 <- 1")
    print(f, g, ax, bx, ay,by, aw, bw, axw, ayw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    y[0] = by * epsy[0]
    w[0] = np.nan
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = np.nan
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[0] + by * epsy[2]
    w[2] = np.nan
    for i in range(3, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        if i % 2 == 0:
            y[i] = ay * y[i - 2] + by * epsy[i]
            w[i] = np.nan
        else:
            y[i] = np.nan
            w[i] = aw * w[i - 2] + axw * f(x[i - 2]) + ayw * g(y[i - 1]) + bw * epsw[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def mediator_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3,3])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1,2] = 2
    unit[2,1] = 1

    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2), (1, -1)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity
        h = identity

    print("Mediator: 0 -> 2 <- 1 <-0")
    print(f, g, ax, bx, ay,by, aw, bw, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    w[1] = np.nan
    for i in range(2, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        if i % 2 == 0:
            w[i] = aw * w[i - 2] + axw * g(x[i - 2]) + ayw * h(y[i - 1]) + bw * epsw[i]
        else:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def cycle_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3, 3])))
    unit[0, 1] = 2
    unit[1, 0] = 2
    unit[0, 2] = 2
    unit[2, 0] = 1
    temporal = dict()
    temporal[0] = [(0, -1), (0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]

    N=N+2
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayx = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set) - 1)]
    else:
        f = identity
        g = identity
        h = identity

    print("Cycle: 1 <-> 0 -> 2")
    print(f, g, ax, bx, ay,by, aw, bw, axy, axw)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])

    x[0] = bx * epsx[0]
    x[1] = ax * x[0] + bx * epsx[1]
    y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    w[1] = np.nan
    x[2] = ax * x[1] + ayx * y[1] + bx * epsx[2]
    y[2] = ay * y[1] + axy * x[1] + by * epsy[2]
    w[2] = aw * w[0] + axw * x[0] + bw * epsw[2]
    for i in range(3, N):
        x[i] = ax * x[i - 1] + ayx * f(y[i - 1]) + bx * epsx[i]
        y[i] = ay * y[i - 1] + axy * g(x[i - 1]) + by * epsy[i]
        if i % 2 == 0:
            w[i] = aw * w[i - 2] + axw * g(x[i - 2]) + bw * epsw[i]
        else:
            w[i] = np.nan

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])

    series = pd.concat([x, y, w], axis=1, sort=False)
    series = series.drop(series.index[[0,1]])
    series = series.reset_index(drop=True)
    return series, unit, temporal



def diamond_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([4,4])
    unit = np.diag(np.diag(np.ones([4,4])))
    unit[0,1] = 2
    unit[1,0] = 1
    unit[0,2] = 2
    unit[2,0] = 1
    unit[1, 3] = 2
    unit[3, 1] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1
    temporal = dict()
    temporal[0] = [(0, -1)]
    temporal[1] = [(1, -1), (0, -1)]
    temporal[2] = [(2, -1), (0, -2)]
    temporal[3] = [(3, -1), (1, -1), (2, -1)]

    N=N+3
    ax = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bx = 0.1
    ay = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    by = 0.1
    aw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bw = 0.1
    az = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bz = 0.1

    axy = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    axw = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ayz = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    awz = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f = functions_set[random.randint(0, len(functions_set)-1)]
        g = functions_set[random.randint(0, len(functions_set)-1)]
        h = functions_set[random.randint(0, len(functions_set)-1)]
        k = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f = identity
        g = identity
        h = identity
        k = identity

    print("Diamond: 3 <- 1 <- 0 -> 2 -> 3")
    print(f, g, h, k, ax, bx, ay,by, aw, bw, az, bz, axy, axw, ayz, awz)
    epsx = np.random.randn(N) ** 3
    epsy = np.random.randn(N) ** 3
    epsw = np.random.randn(N) ** 3
    epsz = np.random.randn(N) ** 3

    x = np.zeros([N])
    y = np.zeros([N])
    w = np.zeros([N])
    z = np.zeros([N])

    x[0] = bx * epsx[0]
    w[0] = np.nan
    x[1] = ax * x[0] + bx * epsx[1]
    # y[1] = ay * y[0] + axy * x[0] + by * epsy[1]
    y[1] = np.nan
    x[2] = ax * x[1] + bx * epsx[2]
    y[2] = ay * y[2] + axy * x[1] + by * epsy[2]
    # w[2] = aw * w[1] + axw * x[0] + bw * epsw[2]
    w[2] = np.nan
    x[3] = ax * x[2] + bx * epsx[3]
    # y[3] = ay * y[1] + axy * x[2] + by * epsy[3]
    y[3] = np.nan
    w[3] = aw * w[1] + axw * x[1] + bw * epsw[3]
    z[3] = az * z[1] + ayz * h(y[2]) + awz * k(w[1]) + bz * epsz[3]
    for i in range(4, N):
        x[i] = ax * x[i - 1] + bx * epsx[i]
        # y[i] = ay * y[i - 1] + axy * f(x[i - 1]) + by * epsy[i]
        # w[i] = aw * w[i - 1] + axw * g(x[i - 2]) + bw * epsw[i]
        if i % 2 == 0:
            y[i] = ay * y[i - 2] + axy * f(x[i - 1]) + by * epsy[i]
            w[i] = np.nan
        else:
            y[i] = np.nan
            w[i] = aw * w[i - 2] + axw * g(x[i - 2]) + bw * epsw[i]
        z[i] = az * z[i - 2] + ayz * h(y[i - 2]) + awz * k(w[i - 1]) + bz * epsz[i]

    x = pd.DataFrame(x, columns=["V1"])
    y = pd.DataFrame(y, columns=["V2"])
    w = pd.DataFrame(w, columns=["V3"])
    z = pd.DataFrame(z, columns=["V4"])

    series = pd.concat([x, y, w, z], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2,3,4]])
    series = series.reset_index(drop=True)
    return series, unit, temporal


def seven_ts_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.array([[1, 1, 0, 0, 0, 0, 0],
                      [2, 1, 1, 0, 0, 0, 0],
                      [0, 2, 1, 1, 0, 0, 0],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 1, 2, 0],
                      [0, 0, 0, 0, 1, 1, 2],
                      [0, 0, 0, 0, 0, 1, 1]])

    temporal = dict()
    temporal[0] = [(0, -1), (1, -1), (5, -1)]
    temporal[1] = [(1, -1), (0, -1), (2, -1)]
    temporal[2] = [(2, -1), (3, -1)]
    temporal[3] = [(3, -1)]
    temporal[4] = [(4, -1), (3, -1)]
    temporal[5] = [(5, -1), (4, -1), (6, -1)]
    temporal[6] = [(6, -1), (5, -1), (1, -1)]

    N=N+3
    # at1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # bt1 = 0.1
    # at2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # bt2 = 0.1
    aa = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ba = 0.1
    ab = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bb = 0.1
    ac = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bc = 0.1
    ad = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bd = 0.1
    ae = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    be = 0.1
    af = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bf = 0.1
    ah = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bh = 0.1

    # atb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # ata = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    abe = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    acf = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ach = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ada = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # ate = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    # atd = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    afb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ahd = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f0 = functions_set[random.randint(0, len(functions_set)-1)]
        f1 = functions_set[random.randint(0, len(functions_set)-1)]
        # f2 = functions_set[random.randint(0, len(functions_set)-1)]
        f3 = functions_set[random.randint(0, len(functions_set)-1)]
        # f4 = functions_set[random.randint(0, len(functions_set)-1)]
        f5 = functions_set[random.randint(0, len(functions_set)-1)]
        # f6 = functions_set[random.randint(0, len(functions_set)-1)]
        f7 = functions_set[random.randint(0, len(functions_set)-1)]
        # f8 = functions_set[random.randint(0, len(functions_set)-1)]
        f9 = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f0 = identity
        f1 = identity
        # f2 = identity
        f3 = identity
        # f4 = identity
        f5 = identity
        # f6 = identity
        f7 = identity
        # f8 = identity
        f9 = identity

    print("7 time series structure with no hidden causes")
    print(f9, f7, f0, f1, f3, f5, ae, be, ab, bb, af, bf, ac, bc, ah, bh, ad, bd, aa, ba, abe, afb, acf, ach, ahd, ada)
    # epst1 = np.random.randn(N) ** 3
    # epst2 = np.random.randn(N) ** 3
    epsa = np.random.randn(N) ** 3
    epsb = np.random.randn(N) ** 3
    epsc = np.random.randn(N) ** 3
    epsd = np.random.randn(N) ** 3
    epse = np.random.randn(N) ** 3
    epsf = np.random.randn(N) ** 3
    epsh = np.random.randn(N) ** 3

    # t1 = np.zeros([N])
    # t2 = np.zeros([N])
    a = np.zeros([N])
    b = np.zeros([N])
    c = np.zeros([N])
    d = np.zeros([N])
    e = np.zeros([N])
    f = np.zeros([N])
    h = np.zeros([N])

    # t1[0] = bt1 * epst1[0]
    # t2[0] = bt2 * epst2[0]
    c[0] = bc * epsc[0]
    f[0] = bf * epsf[0]
    h[0] = bh * epsh[0]
    d[0] = bd * epsd[0]
    a[0] = ba * epsa[0]
    b[0] = bb * epsb[0]
    e[0] = be * epse[0]

    # t1[1] = at1 * t1[0] + bt1 * epst1[1]
    # t2[1] = at2 * t2[0] + bt2 * epst2[1]
    c[1] = ac * c[0] + bc * epsc[1]
    f[1] = af * f[0] + acf * f0(c[0]) + bf * epsf[1]
    h[1] = ah * h[0] + ach * f1(c[0]) + bh * epsh[1]
    d[1] = ad * d[0] + ahd * f3(h[0]) + bd * epsd[1]
    a[1] = aa * a[0] + ada * f5(d[0]) + ba * epsa[1]
    b[1] = ab * b[0] + afb * f7(f[0]) + bb * epsb[1]
    e[1] = ae * e[0] + abe * f9(b[0]) + be * epse[1]

    for i in range(2, N):
        # t1[i] = at1 * t1[i - 1] + bt1 * epst1[i]
        # t2[i] = at2 * t2[i - 1] + bt2 * epst2[i]
        c[i] = ac * c[i - 1] + bc * epsc[i]

        f[i] = af * f[i - 1] + acf * f0(c[i - 1]) + bf * epsf[i]
        h[i] = ah * h[i - 1] + ach * f1(c[i - 1]) + bh * epsh[i]
        d[i] = ad * d[i - 1] + ahd * f3(h[i - 1]) + bd * epsd[i]

        if i % 2 == 0:
            a[i] = aa * a[i - 2] + ada * f5(d[i - 1]) + ba * epsa[i]
            b[i] = ab * b[i - 2] + afb * f7(f[i - 1]) + bb * epsb[i]
            e[i] = np.nan
        else:
            a[i] = np.nan
            b[i] = np.nan
            e[i] = ae * e[i - 2] + abe * f9(b[i - 1]) + be * epse[i]

    a = pd.DataFrame(a, columns=["A"])
    b = pd.DataFrame(b, columns=["B"])
    f = pd.DataFrame(f, columns=["F"])
    c = pd.DataFrame(c, columns=["C"])
    h = pd.DataFrame(h, columns=["H"])
    d = pd.DataFrame(d, columns=["D"])
    e = pd.DataFrame(e, columns=["E"])

    series = pd.concat([e, b, f, c, h, d, a], axis=1, sort=False)
    series = series.drop(series.index[[0, 1, 2]])
    series = series.reset_index(drop=True)
    return series, unit, temporal



def structure_with_2_hidden_var_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.array([[1, 1, 0, 0, 0, 2, 0],
                      [2, 1, 1, 0, 0, 0, 2],
                      [0, 2, 1, 1, 0, 0, 0],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 1, 2, 0],
                      [2, 0, 0, 0, 1, 1, 2],
                      [0, 2, 0, 0, 0, 1, 1]])

    temporal = dict()
    temporal[0] = [(0, -1), (1, -1), (5, -1)]
    temporal[1] = [(1, -1), (0, -1), (2, -1)]
    temporal[2] = [(2, -1), (3, -1)]
    temporal[3] = [(3, -1)]
    temporal[4] = [(4, -1), (3, -1)]
    temporal[5] = [(5, -1), (4, -1), (6, -1)]
    temporal[6] = [(6, -1), (5, -1), (1, -1)]

    N=N+3
    at1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bt1 = 0.1
    at2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bt2 = 0.1
    aa = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ba = 0.1
    ab = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bb = 0.1
    ac = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bc = 0.1
    ad = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bd = 0.1
    ae = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    be = 0.1
    af = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bf = 0.1
    ah = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    bh = 0.1

    atb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ata = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    abe = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    acf = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ach = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ada = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ate = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    atd = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    afb = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    ahd = uniform_with_gap(min_value,max_value, min_gap, max_gap)

    if non_linear:
        f0 = functions_set[random.randint(0, len(functions_set)-1)]
        f1 = functions_set[random.randint(0, len(functions_set)-1)]
        f2 = functions_set[random.randint(0, len(functions_set)-1)]
        f3 = functions_set[random.randint(0, len(functions_set)-1)]
        f4 = functions_set[random.randint(0, len(functions_set)-1)]
        f5 = functions_set[random.randint(0, len(functions_set)-1)]
        f6 = functions_set[random.randint(0, len(functions_set)-1)]
        f7 = functions_set[random.randint(0, len(functions_set)-1)]
        f8 = functions_set[random.randint(0, len(functions_set)-1)]
        f9 = functions_set[random.randint(0, len(functions_set)-1)]
    else:
        f0 = identity
        f1 = identity
        f2 = identity
        f3 = identity
        f4 = identity
        f5 = identity
        f6 = identity
        f7 = identity
        f8 = identity
        f9 = identity

    print("complex structure with two hidden causes")
    # print(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, atb, ata, abe, acf, ach, ada, ate, atd, afb, ahd)
    print(f8, f6, f2, f4, f9, f7, f0, f1, f3, f5, at1, bt1, at2, bt2, ae, be, ab, bb, af, bf, ac, bc, ah, bh, ad, bd,
          aa, ba, ate, atb, atd, ate, abe, afb, acf, ach, ahd, ada)

    epst1 = np.random.randn(N) ** 3
    epst2 = np.random.randn(N) ** 3
    epsa = np.random.randn(N) ** 3
    epsb = np.random.randn(N) ** 3
    epsc = np.random.randn(N) ** 3
    epsd = np.random.randn(N) ** 3
    epse = np.random.randn(N) ** 3
    epsf = np.random.randn(N) ** 3
    epsh = np.random.randn(N) ** 3

    t1 = np.zeros([N])
    t2 = np.zeros([N])
    a = np.zeros([N])
    b = np.zeros([N])
    c = np.zeros([N])
    d = np.zeros([N])
    e = np.zeros([N])
    f = np.zeros([N])
    h = np.zeros([N])

    t1[0] = bt1 * epst1[0]
    t2[0] = bt2 * epst2[0]
    c[0] = bc * epsc[0]
    f[0] = bf * epsf[0]
    h[0] = bh * epsh[0]
    d[0] = bd * epsd[0]
    a[0] = ba * epsa[0]
    b[0] = bb * epsb[0]
    e[0] = be * epse[0]

    t1[1] = at1 * t1[0] + bt1 * epst1[1]
    t2[1] = at2 * t2[0] + bt2 * epst2[1]
    c[1] = ac * c[0] + bc * epsc[1]
    f[1] = af * f[0] + acf * f0(c[0]) + bf * epsf[1]
    h[1] = ah * h[0] + ach * f1(c[0]) + bh * epsh[1]
    d[1] = ad * d[0] + atd * f2(t2[0]) + ahd * f3(h[0]) + bd * epsd[1]
    a[1] = aa * a[0] + ata * f4(t1[0]) + ada * f5(d[0]) + ba * epsa[1]
    b[1] = ab * b[0] + atb * f6(t1[0]) + afb * f7(f[0]) + bb * epsb[1]
    e[1] = ae * e[0] + ate * f8(t2[0]) + abe * f9(b[0]) + be * epse[1]

    for i in range(2, N):
        t1[i] = at1 * t1[i - 1] + bt1 * epst1[i]
        t2[i] = at2 * t2[i - 1] + bt2 * epst2[i]
        c[i] = ac * c[i - 1] + bc * epsc[i]

        f[i] = af * f[i - 1] + acf * f0(c[i - 1]) + bf * epsf[i]
        h[i] = ah * h[i - 1] + ach * f1(c[i - 1]) + bh * epsh[i]
        d[i] = ad * d[i - 1] + atd * f2(t2[i - 1]) + ahd * f3(h[i - 1]) + bd * epsd[i]

        if i % 2 == 0:
            a[i] = aa * a[i - 2] + ada * f5(d[i - 1]) + ba * epsa[i]
            b[i] = ab * b[i - 2] + afb * f7(f[i - 1]) + bb * epsb[i]
            e[i] = np.nan
        else:
            a[i] = np.nan
            b[i] = np.nan
            e[i] = ae * e[i - 2] + abe * f9(b[i - 1]) + be * epse[i]
        #
        # a[i] = aa * a[i - 1] + ata * f4(t1[i - 1]) + ada * f5(d[i - 1]) + ba * epsa[i]
        # b[i] = ab * b[i - 1] + atb * f6(t1[i - 1]) + afb * f7(f[i - 1]) + bb * epsb[i]
        #
        # e[i] = ae * e[i - 1] + ate * f8(t2[i - 1]) + abe * f9(b[i - 1]) + be * epse[i]

    a = pd.DataFrame(a, columns=["A"])
    b = pd.DataFrame(b, columns=["B"])
    f = pd.DataFrame(f, columns=["F"])
    c = pd.DataFrame(c, columns=["C"])
    h = pd.DataFrame(h, columns=["H"])
    d = pd.DataFrame(d, columns=["D"])
    e = pd.DataFrame(e, columns=["E"])

    series = pd.concat([e, b, f, c, h, d, a], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2]])
    series = series.reset_index(drop=True)
    series.index.names = ['time_index']
    return series, unit, temporal


def hidden_generator_diff_sampling_rate(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    unit = np.diag(np.diag(np.ones([3, 3])))
    unit[0, 1] = 2
    unit[1, 0] = 2
    unit[0, 2] = 2
    unit[2, 0] = 1
    unit[1, 2] = 2
    unit[2, 1] = 1
    temporal = dict()
    temporal[0] = [(0, -1), (1, -1)]
    temporal[1] = [(1, -1), (0, -2)]
    temporal[2] = [(2, -1), (0, -1), (1, -1)]

    series, _, _ = diamond_generator_diff_sampling_rate(N=N, non_linear=non_linear, min_value=min_value, max_value=max_value, min_gap=min_gap, max_gap=max_gap)
    series = series.drop(["V1"], axis=1)
    return series, unit, temporal


def complex_7_mooij_dsr(N=1000, non_linear=True, min_value=-1, max_value=1, min_gap=-0.1, max_gap=0.1):
    # unit = np.zeros([7,7])
    unit = np.diag(np.diag(np.ones([7,7])))

    unit[2, 1] = 2
    unit[1, 2] = 1
    unit[2, 3] = 2
    unit[3, 2] = 1
    unit[2, 6] = 2
    unit[6, 2] = 1
    unit[1, 3] = 2
    unit[3, 1] = 1
    unit[1, 5] = 2
    unit[5, 1] = 1
    unit[1, 6] = 2
    unit[6, 1] = 1
    unit[1, 0] = 2
    unit[0, 1] = 1
    unit[3, 5] = 2
    unit[5, 3] = 1
    unit[3, 0] = 2
    unit[0, 3] = 1
    unit[5, 6] = 2
    unit[6, 5] = 1
    unit[5, 4] = 2
    unit[4, 5] = 1
    unit[6, 0] = 2
    unit[0, 6] = 1
    unit[6, 4] = 2
    unit[4, 6] = 1

    temporal = dict()
    temporal[2] = [(2, -1)]
    temporal[1] = [(1, -1), (2, -1)]
    temporal[3] = [(3, -1), (0, -1), (1, -1)]
    temporal[5] = [(5, -1), (1, -1), (3, -1)]
    temporal[6] = [(6, -1), (2, -1), (5, -1)]
    temporal[0] = [(0, -1), (1, -1), (3, -1), (6, -1)]
    temporal[4] = [(4, -1), (1, -1), (5, -1), (6, -1)]

    N=N+5
    a1 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b1 = 0.1
    a2 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b2 = 0.1
    a3 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b3 = 0.1
    a4 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b4 = 0.1
    a5 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b5 = 0.1
    a6 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b6 = 0.1
    a7 = uniform_with_gap(min_value,max_value, min_gap, max_gap)
    b7 = 0.1

    print("Complex structure")
    eps1 = np.random.uniform(low=-0.1, high=0.1, size=N)
    eps2 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps3 = np.random.uniform(low=-1.0, high=1.0, size=N)
    eps4 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps5 = np.random.uniform(low=-0.2, high=0.2, size=N)
    eps6 = np.random.uniform(low=-0.5, high=0.5, size=N)
    eps7 = np.random.uniform(low=-0.3, high=0.3, size=N)

    v1 = np.zeros([N])
    v2 = np.zeros([N])
    v3 = np.zeros([N])
    v4 = np.zeros([N])
    v5 = np.zeros([N])
    v6 = np.zeros([N])
    v7 = np.zeros([N])

    v3[0] = b3 * eps3[0]
    v3[1] = a3 * v3[0] + b3 * eps3[1]
    v2[1] = a2 * v2[0] + np.power(v3[0], 2) + b2 * eps2[1]
    v3[2] = a3 * v3[1] + b3 * eps3[2]
    v2[2] = a2 * v2[1] + np.power(v3[1], 2) + b2 * eps2[2]
    v4[2] = a4 * v4[1] + np.sin(v2[1]) + np.sin(2 * v3[1]) + b4 * eps4[2]
    v3[3] = a3 * v3[2] + b3 * eps3[3]
    v2[3] = a2 * v2[2] + np.power(v3[2], 2) + b2 * eps2[3]
    v4[3] = a4 * v4[2] + np.sin(v2[2]) + np.sin(2 * v3[2]) + b4 * eps4[3]
    v6[3] = a6 * v6[2] + np.sin(v2[2]) + np.cos(2 * v4[2]) + b6 * eps6[3]
    v3[4] = a3 * v3[3] + b3 * eps3[4]
    v2[4] = a2 * v2[3] + np.power(v3[3], 3) + b2 * eps2[4]
    v4[4] = a4 * v4[3] + np.sin(v2[3]) + np.sin(2 * v3[3]) + b4 * eps4[4]
    v6[4] = a6 * v6[3] + np.sin(v2[3]) + np.cos(2 * v4[3]) + b6 * eps6[4]
    v7[4] = a7 * v7[3] + np.cos(v6[3] + v3[3]) + b7 * eps7[4]
    v3[5] = a3 * v3[4] + b3 * eps3[5]
    v2[5] = a2 * v2[4] + np.power(v3[4], 3) + b2 * eps2[5]
    v4[5] = a4 * v4[4] + np.sin(v2[4]) + np.sin(2 * v3[4]) + b4 * eps4[5]
    v6[5] = a6 * v6[4] + np.sin(v2[4]) + np.cos(2 * v4[4]) + b6 * eps6[5]
    v7[5] = a7 * v7[4] + np.cos(v6[4] + v3[4]) + b7 * eps7[5]
    v1[5] = a1 * v1[4] + np.sin(np.power(v4[4], 2)) + np.power(v2[4], 2) + np.cos(v7[4]) + b1 * eps1[5]
    v5[5] = a5 * v5[4] + np.tanh(v6[4] + v7[4] + v2[4]) + b5 * eps5[5]
    for i in range(6, N):
        v3[i] = a3 * v3[i - 1] + b3 * eps3[i]
        v2[i] = a2 * v2[i - 1] + np.power(v3[i-1], 2) + b2 * eps2[i]
        v4[i] = a4 * v4[i - 1] + np.sin(v2[i-1]) + np.sin(2 * v3[i-1]) + b4 * eps4[i]
        v6[i] = a6 * v6[i - 1] + np.sin(v2[i-1]) + np.cos(2 * v4[i-1]) + b6 * eps6[i]
        # v7[i] = a7 * v7[i - 1] + np.cos(v6[i-1] + v3[i-1]) + b7 * eps7[i]
        # v1[i] = a1 * v1[i - 1] + np.sin(np.power(v4[i-1], 2)) + np.power(v2[i-1], 2) + np.cos(v7[i-1]) + b1 * eps1[i]
        # v5[i] = a5 * v5[i - 1] + np.tanh(v6[i-1] + v7[i-1] + v2[i-1]) + b5 * eps5[i]
        if i % 2 == 0:
            v7[i] = a7 * v7[i - 2] + np.cos(v6[i - 1] + v3[i - 1]) + b7 * eps7[i]
            v1[i] = a1 * v1[i - 2] + np.sin(np.power(v4[i - 1], 2)) + np.power(v2[i - 1], 2) + np.cos(v7[i - 2]) + b1 * \
                    eps1[i]
            v5[i] = np.nan
        else:
            v7[i] = np.nan
            v1[i] = np.nan
            v5[i] = a5 * v5[i - 2] + np.tanh(v6[i - 1] + v7[i - 1] + v2[i - 1]) + b5 * eps5[i]


    v3 = pd.DataFrame(v3, columns=["V3"])
    v2 = pd.DataFrame(v2, columns=["V2"])
    v4 = pd.DataFrame(v4, columns=["V4"])
    v6 = pd.DataFrame(v6, columns=["V6"])
    v7 = pd.DataFrame(v7, columns=["V7"])
    v1 = pd.DataFrame(v1, columns=["V1"])
    v5 = pd.DataFrame(v5, columns=["V5"])

    series = pd.concat([v1, v2, v3, v4, v5, v6, v7], axis=1, sort=False)
    series = series.drop(series.index[[0,1,2,3,4, 5]])
    series = series.reset_index(drop=True)
    return series, unit, temporal

if __name__ == "__main__":

    data = np.random.randn(100000) ** 3
    print(np.mean(data))
    print(np.var(data))
    print(np.std(data))
    print(np.sqrt(15))

