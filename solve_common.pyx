from libc.math cimport sin, cos, tan, sqrt
cimport numpy as np
from scipy.optimize import minimize
from array import array
import cython

# compute dot product of two 3-component vectors
@cython.boundscheck(False)
@cython.wraparound(False)
def dot3(list V1, list V2):
    cdef int i
    cdef double result = 0
    for i in range(3):
        result += V1[i] * V2[i]
    return result

# compute square distance between 2 points
@cython.boundscheck(False)
@cython.wraparound(False)
def dist3(list P1, list P2):
    cdef double dx = P1[0] - P2[0]
    cdef double dy = P1[1] - P2[1]
    cdef double dz = P1[2] - P2[2]
    return sqrt(dx * dx + dy * dy + dz * dz)

# compute square distance between 2 points
@cython.boundscheck(False)
@cython.wraparound(False)
def square_dist3(list P1, list P2):
    cdef double dx = P1[0] - P2[0]
    cdef double dy = P1[1] - P2[1]
    cdef double dz = P1[2] - P2[2]
    return dx * dx + dy * dy + dz * dz
    
# transform a 3-component vector using R and T
@cython.boundscheck(False)
@cython.wraparound(False)
def transform3(list R, list V, list T=[0.0, 0.0, 0.0]):
    cdef list result = T[:]
    cdef int i, j
    for i in range(3):
        for j in range(3):
            result[i] += R[i][j] * V[j]
    return result

# general vector transform
@cython.boundscheck(False)
@cython.wraparound(False)
def transform(list M, list V):
    cdef list result = []
    cdef list row
    cdef double rv, vv, sum
    for row in M:
        sum = 0
        for rv, vv in zip(row, V):
            sum += rv * vv
        result.append(sum)
    return result
    
# scale a 3-component vector
@cython.boundscheck(False)
@cython.wraparound(False)
def scale3(double c, list V):
    return [c * V[0], c * V[1], c * V[2]]

# compute difference between 2 points
@cython.boundscheck(False)
@cython.wraparound(False)
def diff3(list V1, list V2):
    cdef double dx = V1[0] - V2[0]
    cdef double dy = V1[1] - V2[1]
    cdef double dz = V1[2] - V2[2]
    return [dx, dy, dz]

# generate rotation matrix using Roe convention
def generate_rot_matrix(double a1, double a2, double a3):
    cdef double ca1 = cos(a1)
    cdef double sa1 = sin(a1)
    cdef double ca2 = cos(a2)
    cdef double sa2 = sin(a2)
    cdef double ca3 = cos(a3)
    cdef double sa3 = sin(a3)
    return [[ca1 * ca2 * ca3 - sa1 * sa3, -ca1 * ca2 * sa3 - sa1 * ca3, ca1 * sa2],
            [sa1 * ca2 * ca3 + ca1 * sa3, -sa1 * ca2 * sa3 + ca1 * ca3, sa1 * sa2],
            [-sa2 * ca3, sa2 * sa3, ca2]]

# optimization wrapper
def optimize(func, X, iter=20000, thres=0.01, args=()):
    # first L-BFGS-B
    res = minimize(func, X, args, method='L-BFGS-B',
                   options={'disp': False, 'maxiter': iter, 'ftol': 1e-20})

    if res.fun < thres or thres == 0:
        # then Nelder-Mead
        X1 = res.x
        res = minimize(func, X1, args, method='Nelder-Mead',
                       options={'disp': False, 'maxiter': iter, 'maxfev': iter, 'xatol': 1e-20})
    return res
