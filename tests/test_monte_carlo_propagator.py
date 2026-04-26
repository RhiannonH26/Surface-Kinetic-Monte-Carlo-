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
    
def test_propagate_one_step_choose_desorption():
    # checks that desorption events are handled correctly
    np.random.seed(67)
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1.0,
            "k_des" : 2.0,
            "k_diffusion" : 1e-3,
            "k_rxn" : 1e-4
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 12.0,
            "k_des" : 4.0,
            "k_rxn" : 1e-4
        }
    }
    grid_vis=np.zeros((2,2)) # unimportant
    # mostly zeros, 2x2 grid with 2 components, 2 rxn types
    rates = np.zeros((2,2,2,2))
    k_indices = {'k_ads': 0, 'k_des': 1}
    # only one site is active
    rates[0,0,0,1] = 2.0 # one O atom has been adsorbed at the (0,0) site
    # the next event should be adsorption, moved forward in time by:
    time = -np.log(np.random.random())/2.0
    rates,updated_time,grid_vis = monte_carlo_propagator.propagate_monte_carlo_one_step(rates, 0, comps_properties,k_indices,grid_vis,seed=67)
    # the next chosen rate should be a desorption event
    # that means ALL desorption events are possible,
    expected_rates = np.zeros((2,2,2,2))
    expected_rates[0,0,0,0] = 1.0
    expected_rates[0,0,1,0] = 12.0
    assert np.allclose(rates,expected_rates) and time == updated_time

def test_propagate_one_step_choose_adsorption():
    # checks that desorption events are handled correctly
    np.random.seed(67)
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1.0,
            "k_des" : 2.0,
            "k_diffusion" : 1e-3,
            "k_rxn" : 1e-4
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 12.0,
            "k_des" : 4.0,
            "k_rxn" : 1e-4
        }
    }
    # mostly zeros, 2x2 grid with 2 components, 2 rxn types
    grid_vis=np.zeros((2,2)) # unimportant
    rates = np.zeros((2,2,2,2))
    k_indices = {'k_ads': 0, 'k_des': 1}
    # only one site is active
    rates[0,0,0,0] = 1.0 # one O atom could adsorb
    rates[0,0,1,0] = 12.0 # one C atom could adsorb
    # the next event should be adsorption, moved forward in time by:
    rates,updated_time,grid_vis = monte_carlo_propagator.propagate_monte_carlo_one_step(rates, 0, comps_properties,k_indices,grid_vis,seed=67)
    # the next chosen rate should be a desorption event
    # oxygen rate should be removed, only have desorption of C
    expected_rates = np.zeros((2,2,2,2))
    expected_rates[0,0,1,1] = 4.0
    assert np.allclose(rates,expected_rates)

def test_grid_vis():
    # checks that desorption events are handled correctly
    np.random.seed(67)
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1.0,
            "k_des" : 2.0,
            "k_diffusion" : 1e-3,
            "k_rxn" : 1e-4
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 12.0,
            "k_des" : 4.0,
            "k_rxn" : 1e-4
        }
    }
    grid_vis=np.zeros((2,2))
    # mostly zeros, 2x2 grid with 2 components, 2 rxn types
    rates = np.zeros((2,2,2,2))
    k_indices = {'k_ads': 0, 'k_des': 1}
    # only one site is active
    rates[0,0,0,0] = 1.0 # one O atom could adsorb
    rates[0,0,1,0] = 12.0 # one C atom could adsorb
    # the next event should be adsorption, moved forward in time by:
    rates,updated_time,grid_vis = monte_carlo_propagator.propagate_monte_carlo_one_step(rates, 0, comps_properties,k_indices,grid_vis,seed=67)
    # the next chosen rate should be a desorption event
    # oxygen rate should be removed, only have desorption of C
    expected_grid_vis = np.zeros((2,2))
    expected_grid_vis[0,0] = 2. # 1 + 1
    assert np.allclose(grid_vis,expected_grid_vis)