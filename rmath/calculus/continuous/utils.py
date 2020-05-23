import numpy

def shift(x, index, offset):
    x_ = numpy.array(x)
    x_[index] +=  offset
    return x_