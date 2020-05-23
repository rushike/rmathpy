import math
import numpy
import itertools

from rmath.calculus.continuous import differentiate
from .utils import get_input_for_optimizer_fn

def fn(x : numpy.ndarray):
    x = get_input_for_optimizer_fn(x)
    return x * x

def fn_g(x : int, order = 1):
    pass


import math


class Adam:
    def __init__(self, fn = None, N = 1, params = {}):
        """Optimizer parameters, default from research
        """
        self.alpha = params.get("alpha", 0.01)
        self.beta_1 = params.get("beta_1", 0.9)
        self.beta_2 = params.get("beta_2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        
        self.fn = fn   # function to optimize
        self.x  = None # initial parameters to start
        self.N  = N    # number of parameters
        self.e  = 0.001
        self.t  = 0

    def set_fn(self, fn):
        self.fn = fn
    
    def set_initial_vector(self, *x):
        if len(x) == 1:
            if isinstance(x[0], tuple) or isinstance(x[0], list):
                self.x = numpy.array(x[0])
                self.N = len(x[0])
                return
        self.x = numpy.array(x)
        self.N = len(x)
            
    def fn_g(self, x, h = 0.5):
        return differentiate(self.fn, x, h = h)

    def apply(self):
        """
        Vectors(numpy.ndarray) to use on start
        """
        x   = numpy.array(self.x)						                        #initialize the vector
        m_t = numpy.zeros(self.N) 
        v_t = numpy.zeros(self.N) 
        t = 0

        while True:					                                                # till it gets converged
            t += 1
            g_t = self.fn_g(x)		                                                # computes the gradient of the stochastic function
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * g_t	                    # updates the moving averages of the gradient
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * (g_t * g_t)	            # updates the moving averages of the squared gradient
            m_cap = m_t / (1 - (self.beta_1 ** t))		                            # calculates the bias-corrected estimates
            v_cap = v_t / (1 - (self.beta_2 ** t))		                            # calculates the bias-corrected estimates
            x_prev = x            
            x = x - (self.alpha * m_cap) / (numpy.sqrt(v_cap) + self.epsilon)	    # updates the parameters            
            if self.__stop(x, x_prev):		                                        # checks if it is converged or not
                break
        self.t = t
        return x

    def __stop(self, x : numpy.ndarray, x_prev : numpy.ndarray):
        if len(x.shape) != len(x_prev.shape) or len(x.shape) > 1 : raise AttributeError(f"x.shape({x.shape}), x_prev.shape({x_prev.shape})  miss matching or not equal to one")
        return abs(sum(x - x_prev) / x.shape[0]) < self.e

    def __str__(self):
        return "Adam Optimizer@{}".format(id(self))