import numpy, random

def test_adam():
    from .adam import Adam, fn
    from .utils import get_input_for_optimizer_fn
    def fn_(x : numpy.ndarray):
        x = get_input_for_optimizer_fn(x)        
        return (x[0]**2) / 100  + (x[1] - 6) ** 2 
    x = (random.randint(1, 100), random.randint(1, 100))
    h = [1, 0.1]
    S = sum(h)
    w = [w_ / S for w_ in h]
    alpha = numpy.average(h, weights=w) 
    adam = Adam(
                fn_,                
                T = 100000,
                params={
                        "alpha" : alpha
                    }
                )
    adam.set_initial_vector(x)
    adam.set_neighbourhood(h)
    print(adam)
    z = adam.apply()
    print(f"t : {adam.t}, x : {x}, z : {z}, fn(x) : {fn_(x)} fn'(x) : {adam.fn_g(x)},  |  fn(z) : {fn_(z)} fn'(z) : {adam.fn_g(z)}")
test_adam()
