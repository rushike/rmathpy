import math
import numpy

def nCr(N, k):
    if N < k : raise AttributeError(f"N > k ? given --> N:{N} < k:{k}")
    return math.factorial(N) // (math.factorial(k) * math.factorial(N- k))

def binomial_coef(N):
    return numpy.array([nCr(N, k) for k in range(N + 1)])