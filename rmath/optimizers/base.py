
class BaseOptimizer:
    def __init__(self, **kwargs ):
        """
            **kwargs : fn, x, h, T
        """
        self.fn = kwargs.get("fn", None)                # function to optimize
        self.x  = kwargs.get("x", numpy.array([0]))     # initial parameters to start
        self.N  = len(self.x)                           # number of parameters
        self.h  = kwargs.get("h", numpy.array([0.1]))   # neighbourhood used to calculate the derivative
        self.t  = 0                                     # loop counter    
        self.T  = kwargs.get("T", 1000)                 # stop the loop if iteration exceeds T, even if convergence is not reached

    def __str__(self, classname = "BaseOptimizer", add_attr = dict()):        
        head    =   "_" * 81 + "\n|%-80s|\n|" % (f"{classname} : ") + "_" * 80 + "|\n"
        content =   "|%-15s : %-62s|\n" % ('fn', self.fn) + \
                    "|%-15s : %-62s|\n" % ('x', self.x) + \
                    "|%-15s : %-62s|\n" % ('N', self.N) + \
                    "|%-15s : %-62s|\n" % ('h', self.h) + \
                    "|%-15s : %-62s|\n" % ('T', self.T)
        footer  =   "|" + "_" * 80 + "|"

        for k, v in add_attr.items():
            if k in ["-"] and v:
                content += "|%-15s : %-62s|\n" % (k * 15, k * 62)
                continue
            content += "|%-15s : %-62s|\n" % (k, str(v))
        return head + content + footer
    
    def __repr__(self):
        return self.__str__()