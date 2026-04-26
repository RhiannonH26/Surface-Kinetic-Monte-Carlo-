import numpy as np
from kinetic_monte_carlo import run_kmc

def test_kindices_diff_len():
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1e-9,
            "k_des" : 1e-10,
            "k_rxn" : 1e-4,
            "k_diffusion" : 1e-3,
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1e-4,
            "k_des" : 1e-8,
            "k_rxn" : 1e-4
        }
    }
    exp_k_indices = {'k_ads' : 0, 'k_des': 1, 'k_rxn': 2, 'k_diffusion': 3}
    acc_k_indices = run_kmc.generate_k_indices(comps_properties)
    assert exp_k_indices == acc_k_indices