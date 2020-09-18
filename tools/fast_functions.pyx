import numpy as np
import cython
cimport numpy as np
from libc.math cimport abs
from tigramite.independence_tests import CMIknn

def _num_neighbors_xy_cython_old(double[:,:] v, int N, int dim_x, int dim_y, double[:] epsarray, int k):
    cdef int[:] nx = np.zeros(N, dtype='int32')
    cdef int[:] ny = np.zeros(N, dtype='int32')
    cdef int i, j, d
    cdef double dy, dx

    for i in range(N):
        for j in range(N):
            dx = 0.
            for d in range(0, dim_y):
                dx = max(abs(v[i, d] - v[j, d]), dx)
            dy = 0.
            for d in range(dim_y, dim_x+dim_y):
                dy = max(abs(v[i, d] - v[j, d]), dy)

            # For no conditions, kz is counted up to T
            if dx < epsarray[i]:
                nx[i] += 1
            if dy < epsarray[i]:
                ny[i] += 1
    return nx, ny

def _num_neighbors_x_cython(double[:,:] v, int N, int dim_x, double[:] epsarray, int k):
    cdef int[:] nx = np.zeros(N, dtype='int32')
    cdef int i, j, d
    cdef double dy, dx

    for i in range(N):
        for j in range(N):
            dx = abs(v[i, 0] - v[j, 0])
            for d in range(1, dim_x):
                dx = max(abs(v[i, d] - v[j, d]), dx)

            # For no conditions, kz is counted up to T
            if dx < epsarray[i]:
                nx[i] += 1
    return nx

def _num_neighbors_xy_cython(double[:,:] v, int N, int dim_x, int dim_y, double[:] epsarray, int k):
    cdef int[:] nx = np.zeros(N, dtype='int32')
    cdef int[:] ny = np.zeros(N, dtype='int32')
    cdef int i, j, d
    cdef double dy, dx

    for i in range(N):
        for j in range(N):
            dx = abs(v[i, 0] - v[j, 0])
            for d in range(1, dim_x):
                dx = max(abs(v[i, d] - v[j, d]), dx)
            dy = abs(v[i, dim_x] - v[j, dim_x])
            for d in range(dim_x+1, dim_x+dim_y):
                dy = max(abs(v[i, d] - v[j, d]), dy)

            # For no conditions, kz is counted up to T
            if dx < epsarray[i]:
                nx[i] += 1
            if dy < epsarray[i]:
                ny[i] += 1
    return nx, ny

def _num_neighbors_xyz_cython(double[:,:] v, int N, int dim_x, int dim_y, int dim_z, double[:] epsarray, int k):
    cdef int[:] nxz = np.zeros(N, dtype='int32')
    cdef int[:] nyz = np.zeros(N, dtype='int32')
    cdef int[:] nz = np.zeros(N, dtype='int32')
    cdef int i, j, d
    cdef double dy, dx, dz

    for i in range(N):
        for j in range(N):

            dz = abs(v[i, dim_x+dim_y] - v[j, dim_x+dim_y])
            for d in range(dim_x+dim_y+1, dim_x+dim_y+dim_z):
                dz = max(abs(v[i, d] - v[j, d]), dz)

            if dz < epsarray[i]:
                nz[i] += 1

                dx = abs(v[i, 0] - v[j, 0])
                for d in range(1, dim_x):
                    dx = max(abs(v[i, d] - v[j, d]), dx)

                dy = abs(v[i, dim_x] - v[j, dim_x])
                for d in range(dim_x+1, dim_x+dim_y):
                    dy = max(abs(v[i, d] - v[j, d]), dy)

                if dx < epsarray[i]:
                    nxz[i] += 1
                if dy < epsarray[i]:
                    nyz[i] += 1
    return nxz, nyz, nz

