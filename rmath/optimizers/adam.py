import math
import numpy
import itertools

from rmath.calculus.continuous import differentiate
from .utils import get_input_for_optimizer_fn
from .base import BaseOptimizer

def fn(x : numpy.ndarray):
    x = get_input_for_optimizer_fn(x)
    return x * x

def fn_g(x : int, order = 1):
    pass


import math


class Adam(BaseOptimizer):
    def __init__(self, fn = None, params = {}, **kwargs):
        ########################################################################
        # Adam parameters                                                      #
        ########################################################################
        self.alpha = params.get("alpha", 0.01)
        self.beta_1 = params.get("beta_1", 0.9)
        self.beta_2 = params.get("beta_2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        
        ########################################################################
        # Non Adam parameters                                                  #
        ########################################################################
        self.fn     = fn                                    # function to optimize
        self.x      = kwargs.get("x", numpy.array([0]))     # initial parameters to start
        self.N      = len(self.x)                           # number of parameters
        self.h      = kwargs.get("h", numpy.array([0.1]))   # neighbourhood used to calculate the derivative
        self.fn_g   = self.fn_g_
        self.t      = 0                                     # loop counter    
        self.T      = kwargs.get("T", 1000)                 # stop the loop if iteration exceeds T, even if convergence is not reached

    def set_fn(self, fn):
        self.fn = fn
    
    def set_initial_vector(self, *x):
        if len(x) == 1:
            if isinstance(x[0], tuple) or isinstance(x[0], list):
                self.x = numpy.array(x[0])
                self.N = len(x[0])
                h_ = numpy.array([self.h[0] for _ in range(self.N)])
                self.h = h_
                return
        self.x = numpy.array(x)
        self.N = len(self.x)
        h_ = numpy.array([self.h[0] for _ in range(self.N)])
        self.h = h_
            
    def set_fn_g(self, fn_g):
        self.fn_g = fn_g
    
    def set_neighbourhood(self, h : numpy.ndarray):
        if len(h) != self.N: raise AttributeError(f"Neighbourhood and set, length mismath, len(h)@[{len(h)}] != self.N@[{self.N}]")        
        self.h = numpy.array(h)

    def fn_g_(self, x, h = 0.5):        
        return differentiate(self.fn, x, h = h)

    def apply(self):
        """
        Vectors(numpy.ndarray) to use on start
        """
        x   = numpy.array(self.x, dtype='float')						                        #initialize the vector
        m_t = numpy.zeros(self.N, dtype='float') 
        v_t = numpy.zeros(self.N, dtype='float') 
        t = 0

        while t < self.T:					                                                # till it gets converged
            t += 1
            g_t = self.fn_g(x)		                                                # computes the gradient of the stochastic function            
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * g_t	                    # updates the moving averages of the gradient
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * (g_t * g_t)	            # updates the moving averages of the squared gradient
            m_cap = m_t / (1 - (self.beta_1 ** t))		                            # calculates the bias-corrected estimates
            v_cap = v_t / (1 - (self.beta_2 ** t))		                            # calculates the bias-corrected estimates
            x_prev = x
            x = x - (self.alpha * m_cap) / (numpy.sqrt(v_cap) + self.epsilon)	    # updates the parameters            
            if self.__stop(x, x_prev) and t > 25:		                                        # checks if it is converged or not
                break
        self.t = t
        return x

    def __stop(self, x : numpy.ndarray, x_prev : numpy.ndarray):
        if len(x.shape) != len(x_prev.shape) or len(x.shape) > 1 or len(x) > len(self.h) : raise AttributeError(f"x.shape({x.shape}), x_prev.shape({x_prev.shape})  miss matching or not equal to one")
        for a, b, c in zip(x, x_prev, self.h):
            if abs(a - b) > c : return False
        return True

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return super().__str__(
                    classname="Adam Optimizer@{}".format(id(self)), 
                    add_attr = {
                            "-"         : True,
                            "alpha"     : self.alpha,
                            "beta_1"    : self.beta_1,
                            "beta_2"    : self.beta_2,
                            "epsilon"   : self.epsilon,
                        }
                )