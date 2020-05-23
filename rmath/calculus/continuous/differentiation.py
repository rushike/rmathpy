import numpy
from .utils import shift

def differentiate(fn, x, order = 1, h = 1, mod = False):
    x = numpy.array(x)
    diff = numpy.array([( ( fn( shift(x, i, h) ) - fn(x) ) / h ) for i in range(x.shape[0])])
    return diff
