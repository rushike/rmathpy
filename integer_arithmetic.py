import numpy
from fractions import Fraction

def matmulpoly(a, b):
  N = len(a)
  M = len(b)
  res = numpy.zeros((N, N), dtype = 'object') + Fraction()
  cor = numpy.zeros(2*N + 1, dtype = 'object') + Fraction()
  for i in range(N):
    for j in range(N):
#        print(type(a[i]), " oooooooooo ", typace(b[i]))
       rt = a[i] * b[j]
       cor[i + j] += rt 
#        print(i, j , " ------ ", rt)
       res[i][j] = rt       
  return cor[:N]

a = numpy.array([2, 1], dtype = 'object')
b = numpy.array([2, 1], dtype = 'object')
print(a)
print(b)
matmulpoly(a, b)

def IntegerSeries(n, r = None):
  if not r : r = n
  if r == 1:
    a = numpy.zeros(n, dtype = 'object') + Fraction()
    a[0] = Fraction(1, 1)
    return a
  else :
    a = numpy.power(Fraction(1, r), numpy.arange(n, dtype = 'object'), dtype = 'object')
    b = IntegerSeries(n, r - 1)
    # return matmulpoly(a, b)
    return matmulpoly(a, b)