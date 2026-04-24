import numpy as np
from kinetic_monte_carlo import monte_carlo_propagator

def test_choose_rate_large_samples():
    # with large statistical sampling, should get the distribution back
    rates = np.array([20.0,50.0,30.0])
    init_vals = np.zeros((3,))
    big_len = 1000000
    for _ in range(big_len):
        index = monte_carlo_propagator._choose_random_rate(rates)
        init_vals[index] += 1
    probabilities = init_vals/big_len*100
    assert np.allclose(rates,probabilities,1e-2)
