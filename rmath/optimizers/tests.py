import numpy

def test_adam():
    from .adam import Adam, fn
    from .utils import get_input_for_optimizer_fn
    def fn_(x : numpy.ndarray):
        x = get_input_for_optimizer_fn(x)
        # x^2 * (y -6) ^ 4
        return x[0] ** 2 + (x[1] - 6) ** 4
    adam = Adam(fn_)
    x = (7, 7)
    adam.set_initial_vector(x)
    z = adam.apply()
    print(f"x : {x}, z : {z}, t : {adam.t}")
test_adam()
