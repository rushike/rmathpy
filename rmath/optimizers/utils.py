from collections.abc import Iterable
import numpy


def get_input_for_optimizer_fn(x):
    """x may be int, float, array, tuple

    Arguments:
        x {int | float | numpy.ndarray | tuple | list} -- [description]
       
    Returns:
        [numpy.ndarray] -- [description]
    """
    if not isinstance(x, Iterable):
        return numpy.array([x])
    else:
        return numpy.array(x)    
    