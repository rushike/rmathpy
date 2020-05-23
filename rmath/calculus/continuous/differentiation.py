import numpy

def differentiate(fn, x, order = 1, h = 1):
    a = fn(x)
    b = fn(x + h)
    diff = (b - a) / h
    return diff