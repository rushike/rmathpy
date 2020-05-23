import numpy
from fractions import Fraction
from numbers import *


class Polynomial(numpy.polynomial.Polynomial):
    def __init__(self, y):
        super().__init__(y)
        pass

class SeriesF(Polynomial):
    def __init__(self, y, length = None):
        self.i = 0
        super().__init__(numpy.array(y) + Fraction())
        self.length = length if length else len(y)
        pass
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i >= self.length: 
            self.i = 0
            raise StopIteration
        self.i += 1
        return self.coef[self.i - 1]

class IntegerSeries(SeriesF):
    def __init__(self, n, r = None):
        a = numpy.zeros(n, dtype = 'object') + Fraction()
        a[0] = Fraction(1, 1)
        self.s = 1
        super().__init__(numpy.array(a) + Fraction())
        self.coef =  self.__built(n, r)        

    def __built(self, n, r = None):
        if not r : r = n
        if r == self.s:
            return self.coef
        else :
            a = numpy.power(Fraction(1, r), numpy.arange(n, dtype = 'object'), dtype = 'object')
            b = self.__built(n, r - 1)
            # return matmulpoly(a, b)
            return self.matmulpoly(a, b)

    def matmulpoly(self, a, b):
        N = len(a)
        M = len(b)
        res = numpy.zeros((N, N), dtype = 'object') + Fraction()
        cor = numpy.zeros(2*N + 1, dtype = 'object') + Fraction()
        for i in range(N):
            for j in range(N):
                rt = a[i] * b[j]
                cor[i + j] += rt 
                res[i][j] = rt       
        return cor[:N]
    
    def nums(self):
        return numpy.array([v._numerator for v in self], dtype = 'object')