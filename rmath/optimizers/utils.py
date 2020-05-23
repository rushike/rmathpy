import numpy

def get_input_for_optimizer_fn(x):
    """x may be int, float, array, tuple

    Arguments:
        x {int | float | numpy.ndarray | tuple | list} -- [description]
       
    Returns:
        [numpy.ndarray] -- [description]
    """
    if isinstance(x, int) or isinstance(x, float):
        return numpy.array([x])
    if isinstance(x, tuple) or isinstance(x, list) or isinstance(x, numpy.ndarray):
        return numpy.array(x)
    return None