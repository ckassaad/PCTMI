"""

  Module for computing distances between timeseries.
  Available distances includes cort, dtw, dtw_cort.
  It requires numpy.

"""
# $Id: distancesV1.py 3903 2016-08-22 14:12:33Z fthollard/S.Alkhoury $

import numpy as np
from cffi import FFI
import platform
import sys
import os.path
import tools.DistanceObject as Do


precision = -0.00000000001


def load_shared():
    """
    Loads a wrapper over the shared library). The shared library is built
    using the command line python3 setup.py build_ext
    The library path should include the path where the shared library is.

    """

    ffi = FFI()
    ffi.cdef("""
    int get_idx(int var_idx, int time_idx, int nbvar);
    double compute_diff(double *x, double *y, int t_x, int t_y, int nbvar);
    void compute_slope(double *x, int start_x, int end_x, int nb_var, double *res);
    void swap_pointers(double **x, double **y);
    void print_multivariate(double *x, int nbvar, int timelen);
    void twed(double *x, double *y, int size_x, int size_y, double *distances);
    void dtw(double *x, double *y, int size_x, int size_y, int nbvar, double *path, double* distances, int window);
    void dtw_weighted(double *x, double *y, int size_x, int size_y, int nbvar, double* distances, double* weight, int window);
    void dtw_given_path(double* temporal_distances, int* path, double* weight, int path_length, double* distances);
    double cort_simple(double *x, double *y, const unsigned time_start, const unsigned time_end,  const unsigned nb_var);
    double cort(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k);
    void cort_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window);
    double cort_dtw(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k);
    void cort_dtw_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window);
    """)

    my_path = os.path.abspath(os.path.dirname(__file__))
    if platform.system() == 'Linux':
        _lib_name = 'cort/distances.cpython-34m.so'
    elif platform.system() == 'Darwin':
        _lib_name = 'cort/distances.cpython-35m-darwin.so'
    else:
        raise RuntimeError('unhandled system {}, cannot load shared library'.format(platform.system()))

    return ffi, ffi.dlopen('/'.join([my_path, _lib_name]))

__ffi, __dllib = load_shared()


def twed(series_a=None, series_b=None):

    """
    Computes the dynamic time wrapping edit distance between series_a and series_b.

    :param series_a: the first series
    :param series_b: the second series
    """
    len_a, len_b = 0, 0
    a_ptr, b_ptr = None, None
    if series_a is not None:
        nb_var = len(series_a.shape)
        if nb_var > 1:
            series_a = series_a.T
            series_b = series_b.T

        len_a = len(series_a)
        len_b = len(series_b)

        a_ptr = __ffi.cast("double*", series_a.ctypes.data)
        b_ptr = __ffi.cast("double*", series_b.ctypes.data)

        if (len_a == 0) or (len_b == 0):
            return sys.float_info.max

        # maximum potential size of the path

        assert len_a >= 0, "time_start_a should be positive (actual value={})".format(len_a)
        assert len_b >= 0, "time_start_b should be positive (actual value={})".format(len_b)

    dist_obj = Do.DistanceObject()
    distances = np.zeros(1, dtype=np.float64)
    d_ptr = __ffi.cast("double*", distances.ctypes.data)
    __dllib.twed(a_ptr, b_ptr, len_a, len_b, d_ptr)
    if dist_obj.additive_dist < precision:
        raise RuntimeError()
    dist_obj.additive_dist = d_ptr[0]
    dist_obj.average_dist_1 = d_ptr[0]
    dist_obj.average_dist_2 = d_ptr[0]

    return dist_obj


def dtw(series_a=None, series_b=None, weight=None, get_path=False, path=None, window=None):

    """
    Computes the dynamic time wrapping distance between (multivariate )series_a and series_b.

    :param series_a: the first series
    :param series_b: the second series
    :param weight: weight vector to use (related to series_a)
    :param get_path: return the path or not
    :param path: Given path to be used in the distance computation
    :param window: window look-ahead for DTW, CorT and CorT_DTW

    :note: The local difference between two points of the time series is the sum of the absolute
    differences of the variables.
    """
    len_a, len_b, nb_var = 0, 0, 0
    a_ptr, b_ptr = None, None
    if series_a is not None:
        nb_var = len(series_a.shape)
        if nb_var > 1:
            series_a = series_a.T
            series_b = series_b.T

        len_a = len(series_a)
        len_b = len(series_b)

        a_ptr = __ffi.cast("double*", series_a.ctypes.data)
        b_ptr = __ffi.cast("double*", series_b.ctypes.data)

        if (len_a == 0) or (len_b == 0):
            return sys.float_info.max

        # maximum potential size of the path

        assert len_a >= 0, "time_start_a should be positive (actual value={})".format(len_a)
        assert len_b >= 0, "time_start_b should be positive (actual value={})".format(len_b)

    dist_obj = Do.DistanceObject()
    temp_window = len(weight) if series_a is None else len(series_a)
    if window is None:
        window = temp_window - 1
    else:
        if 0 < window < 1:
            window = round(temp_window * window) - 1

    if series_a is not None:
        if window < abs(len(series_a) - len(series_b)):
            raise AttributeError('window size is too small to calculate DTW!')

    if window < 0:
        window = 0

    if weight is None:
        all_path = np.zeros((len_a + len_b) * 3, dtype=np.float64)
        distances = np.zeros(3, dtype=np.float64)
        d_ptr = __ffi.cast("double*", distances.ctypes.data)
        p_ptr = __ffi.cast("double*", all_path.ctypes.data)
        __dllib.dtw(a_ptr, b_ptr, len_a, len_b, nb_var, p_ptr, d_ptr, window)
        dist_obj.additive_dist = d_ptr[0]
        dist_obj.average_dist_1 = d_ptr[1]
        dist_obj.average_dist_2 = d_ptr[2]
        if dist_obj.additive_dist < precision or dist_obj.average_dist_1 < precision \
                or dist_obj.average_dist_2 < precision:
            raise RuntimeError()
        if get_path:
            path_size = int(p_ptr[0])
            all_path = [(int(p_ptr[k]), int(p_ptr[k + 1]), p_ptr[k + 2]) for k in range(1, path_size, 3)]
            all_path.reverse()
            all_path = np.array(all_path, dtype=np.float64)
            dist_obj.path = all_path
    else:
        w_ptr = __ffi.cast("double*", weight.ctypes.data)
        if path is not None:
            indexes = path[:, 0]
            temporal_differences = path[:, 1]
            path_length = len(indexes)
            p_ptr = __ffi.new("int[" + str(path_length) + "]")
            for ind, v in enumerate(indexes):
                p_ptr[ind] = int(v)
            t_ptr = __ffi.cast("double*", temporal_differences.ctypes.data)
            distances = np.zeros(2, dtype=np.float64)
            d_ptr = __ffi.cast("double*", distances.ctypes.data)
            __dllib.dtw_given_path(t_ptr, p_ptr, w_ptr, path_length, d_ptr)
            dist_obj.additive_dist = d_ptr[0]
            dist_obj.average_dist_1 = d_ptr[1]
            if dist_obj.additive_dist < precision or dist_obj.average_dist_1 < precision:
                raise RuntimeError()
        else:
            distances = np.zeros(2, dtype=np.float64)
            d_ptr = __ffi.cast("double*", distances.ctypes.data)
            __dllib.dtw_weighted(a_ptr, b_ptr, len_a, len_b, nb_var, d_ptr, w_ptr, window)
            dist_obj.additive_dist = d_ptr[0]
            dist_obj.average_dist_1 = d_ptr[1]
            if dist_obj.additive_dist < precision or dist_obj.average_dist_1 < precision:
                raise RuntimeError()
    return dist_obj


def cort(series_a, series_b, simple=True):
    """
    Computes the tdw cort distance between series_a and series_b.

    :param series_a: the first series
    :param series_b: the second series
    :param weight: weight vector to use (related to series_a)
    :param get_path: return the path or not
    :param path: Given path to be used in the distance computation
    :param interval: time interval considered for series_a (full interval if None). Default=None.
    :type interval: a tuple that represents the interval (start included, end excluded) \
                    in the scale of [0, 100] interval are re-normalized for each series.
    :param window: window look-ahead for DTW, CorT and CorT_DTW
    :return: the distance and optional path
    :rtype: float
    """

    nb_var = len(series_a.shape)
    if nb_var > 1:
        series_a = series_a.T
        series_b = series_b.T

    len_a = len(series_a)
    len_b = len(series_b)

    time_start_a = 0
    time_end_a = len_a
    time_start_b = 0
    time_end_b = len_b

    if time_start_a == time_end_a or time_start_b == time_end_b:
        return sys.float_info.max
    assert time_start_a >= 0,\
        "time_start_a should be positive (actual value={})".format(time_start_a)
    assert time_start_b >= 0, \
        "time_start_b should be positive (actual value={})".format(time_start_b)
    assert time_end_a <= len_a, \
        "time_end_a should not be larger than len(serie_a) {} vs {}, {})".format(time_end_a, len_a, series_a)
    assert time_end_b <= len_b, \
        "time_end_b should not be larger than len(serie_b) {} vs {})".format(time_end_b, len_b)

    a_ptr = __ffi.cast("double*", series_a.ctypes.data)
    b_ptr = __ffi.cast("double*", series_b.ctypes.data)

    if simple:
        res = __dllib.cort_simple(a_ptr, b_ptr, 0, len_a, nb_var)
    else:
        res = __dllib.cort(a_ptr, b_ptr, 0, 0, len_a, len_b, nb_var, 0)
    return res


def cort_dtw(series_a, series_b, weight=None, get_path=False, interval=None, path=None, window=None):
    """
    Computes the cort_dtw distance between series_a and series_b.

    :param series_a: the first series
    :param series_b: the second series
    :param weight: weight vector to use (related to series_a)
    :param get_path: return the path or not
    :param path: Given path to be used in the distance computation
    :param interval: time interval considered for serie_a (full interval if None). Default=None.
    :type interval: a tuple that represents the interval (start included, end excluded) \
                    in the scale of [0, 100] interval are re-normalized for each series.
    :param window: window look-ahead for DTW, CorT and CorT_DTW
    :return: the distance and optional path
    :rtype: float
    """

    nb_var = len(series_a.shape)
    if nb_var > 1:
        series_a = series_a.T
        series_b = series_b.T

    len_a = len(series_a)
    len_b = len(series_b)

    time_start_a = 0 if interval is None else int(interval[0]*len_a/100)
    time_end_a = len_a if interval is None else int(interval[1]*len_a/100)
    time_start_b = 0 if interval is None else int(interval[0]*len_b/100)
    time_end_b = len_b if interval is None else int(interval[1]*len_b/100)

    if time_start_a == time_end_a or time_start_b == time_end_b:
        return sys.float_info.max
    assert time_start_a >= 0,\
        "time_start_a should be positive (actual value={})".format(time_start_a)
    assert time_start_b >= 0, \
        "time_start_b should be positive (actual value={})".format(time_start_b)
    assert time_end_a <= len_a, \
        "time_end_a should not be larger than len(serie_a) {} vs {}, {})".format(time_end_a, len_a, series_a)
    assert time_end_b <= len_b, \
        "time_end_b should not be larger than len(serie_b) {} vs {})".format(time_end_b, len_b)

    a_ptr = __ffi.cast("double*", series_a.ctypes.data)
    b_ptr = __ffi.cast("double*", series_b.ctypes.data)
    dist_obj = Do.DistanceObject()
    temp_window = len(weight) if series_a is None else len(series_a)
    if window is None:
        window = temp_window - 2
    else:
        if 0 < window < 1:
            window = round(temp_window * window) - 2

    if series_a is not None:
        if window < abs(len(series_a) - len(series_b)):
            raise AttributeError('window size is too small to calculate CorT_DTW!')

    if window < 0:
        window = 0

    all_path = np.zeros((len_a + len_b) * 3, dtype=np.float64)
    p_ptr = __ffi.cast("double*", all_path.ctypes.data)
    distances = np.zeros(3, dtype=np.float64)
    d_ptr = __ffi.cast("double*", distances.ctypes.data)
    __dllib.cort_dtw_window_path(a_ptr, b_ptr, 0, 0, len_a, len_b, nb_var, 0, p_ptr, d_ptr, window)
    dist_obj.additive_dist = d_ptr[0]
    dist_obj.average_dist_1 = d_ptr[1]
    dist_obj.average_dist_2 = d_ptr[2]
    if dist_obj.additive_dist < precision or dist_obj.average_dist_1 < precision \
            or dist_obj.average_dist_2 < precision:
        raise RuntimeError()
    if get_path:
        path_size = int(p_ptr[0])
        all_path = [(int(p_ptr[k]), int(p_ptr[k + 1]), p_ptr[k + 2]) for k in range(1, path_size, 3)]
        all_path.reverse()
        all_path = np.array(all_path, dtype=np.float64)
        dist_obj.path = all_path
    return dist_obj



if __name__ == "__main__":
    from data.sim_data import generate_fork, generate_v_structure, generate_mediator, \
        generate_v_structure_ub, generate_fork_ub, generate_diamond, generate_diamond_ub, generate_v_structure_dsr, \
        generate_fork_dsr
    import time

    get_data = {"fork": generate_fork, "v_structure": generate_v_structure, "mediator": generate_mediator,
                "diamond": generate_diamond, "fork_ub": generate_fork_ub, "v_structure_ub": generate_v_structure_ub,
                "diamond_ub": generate_diamond_ub, "v_structure_dsr": generate_v_structure_dsr,
                "fork_dsr": generate_fork_dsr}

    #######
    data_name = "fork"
    scale = True
    #######

    s = time.time()

    order = 3
    n_samples = 5000

    # method = "cti"
    main_method = "mi"
    col1 = 0
    col2 = 2

    result = []
    for it in range(100):
        print("iteration: "+str(it))
        if scale:
            data = get_data[data_name](n_samples)
            data -= data.min()
            data /= data.max()

        # ami = auto_mi(data[data.columns[col1]])
        # print(ami)
        # print(np.mean(auto_mi(data[data.columns[0]])[1:]))
        # print(avg_auto_mi(data[data.columns[col1]]))
        # cami = conditional_auto_mi(data[data.columns[col1]])
        # print(cami)

        if main_method == "mi":
            res = cort(data[data.columns[col1]].values, data[data.columns[col2]].values)
            print("mi: " + str(res))
            result.append(res)
