import numpy as np
from kinetic_monte_carlo import monte_carlo_propagator

def test_choose_rate_zeros():
    # should not have any choices of zero rates, provided that the search sort
    # chooses the first appearance (it should because side='left' is default)
    rates = np.array([20.0,0.0,0.0,70.0])
    init_vals = np.zeros((4,))
    big_len = 10000
    for _ in range(big_len):
        index = monte_carlo_propagator._choose_random_rate(rates)
        init_vals[index] += 1
    zero_rates = init_vals[1:3]
    assert np.allclose(zero_rates,np.zeros((2,)),1e-6)

def test_choose_rates_unflatten():
    # actually tests the rates
    rates = np.zeros((2,2,5,2)) # 2x2 grid with 5 components and 2 rates
    rates[0,0,4,1] = 1.0
    index = monte_carlo_propagator._choose_random_rate(rates)
    assert np.all(index==np.array([0,0,4,1]))

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

def test_count_number_rxn_types():
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1e-9,
            "k_des" : 1e-10,
            "k_diffusion" : 1e-3,
            "k_rxn" : 1e-4
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1e-4,
            "k_des" : 1e-8,
            "k_rxn" : 1e-4
        }
    }
    num = monte_carlo_propagator._count_number_rxn_types(comps_properties)
    assert 4 == float(num)
