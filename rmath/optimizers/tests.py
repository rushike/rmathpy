import numpy

def test_adam():
    from .adam import Adam, fn
    # from .utils import get_input_for_optimizer_fn
    # def fn_(x : numpy.ndarray):
    #     x = get_input_for_optimizer_fn(x)
    # return x * x
    adam = Adam(fn)
    x = (7, 7)
    adam.set_initial_vector(x)
    z = adam.apply()
    print(f"x : {x}, z : {z}, t : {adam.t}")
test_adam()
