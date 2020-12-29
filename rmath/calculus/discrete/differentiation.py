import numpy

from rmath.basic.functions.combinatorics import nCr

def derivative_kernel(order, cls):
    if not cls : cls = int
    return numpy.array([cls((-1) ** k * nCr(N = order, k = k)) for k in range(order + 1)])

def n_derivatives(y, orders = [0], axis = 0):
    pass