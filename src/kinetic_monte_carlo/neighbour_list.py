import numpy as np
from typing import Callable

def generate_rate_const_initial_list(N_grid: float,
                                    comps_properties: dict):
    """Calculates the initial rate constant list for a surface approximated by
    a 2D N_grid x N_grid surface. All surface points are assumed to be the same
    and are a location for a particle to adsorb. This assumes that the surface 
    starts out clean, i.e. no desorption events are possible.

    The list is contains a series of dictionaries specifying the following:
    ```
    {   
        "i" : list(int) or None, # None if exists as a gas in state i, the initial state
        "j" : list(int) or None, # None if exists as a gas in state j, the final state
        "i_atom": str, # identity of atom in state i, might later consider j_atom for reactions
    }
    ```

    The second output is a np.array that contains a list of the numerical rate
    constants that has an index that matches the description above.

    Parameters
    ----------
    N_grid : int
        Number of surface locations along one dimension of the surface.
    comps_properties : list(dict)
        A list of dictionaries containing the properties of each element 
        in each dictionary. It must contain the following properties for each 
        keyword: the element specified with "atom", "partial_pressure" in atm, 
        "mass" in u (atomic units), and "k_ads" the rate constant in units of s^-1. 
        Note that the chosen name or "key" for each component in the first level 
        of the dictionary is unimportant. For example, one element of this list 
        could be:
    
    ```
    {   "atom" : str, # atom identity / element
        "partial_pressure" : float, # partial pressure of the component
        "mass" : float, # mass in u (g/mol) of the component
        "k_ads" : float, # adsorption rate constant
        ...
    }
    ```
    
    Returns
    ----------
    list(dict)
        The list of rate constants to pick from with each element containing a 
        dictionairy describing the rate constant properties.
    np.ndarray
        The numerical value of the rate constants.
    """
    rate_constant_properties = []
    rates = np.zeros((N_grid,N_grid)) # sum over all k
    for k in range(len(comps_properties)):
        for l in range(N_grid):
            for w in range(N_grid):
                myk = comps_properties[k].get("k_ads")
                mydict = {"i": None,
                          "j": [l,w],
                          "i_atom": comps_properties[k].gets("atom")}
                rate_constant_properties.append(mydict)
                rates[l,w] = myk
    # note could have stored things a bit differently but this is how i chose to do it
    return rate_constant_properties, rates.squeeze()

def propagate_monte_carlo_one_step(rate_constant_list: list,
                                   rates: float,
                                   current_time: float,
                                   ):
    """Propagates the kinetic monte carlo algorithm in time by taking one step.
    
    The algorithm works by randomly choosing a rate constant weighted by the size of the
    rate constant and then propagating in time proportional to the inverse of the rate
    constant multiplied by a random number to obey the Poisson distribution (i.e. could
    leave faster than expected).

    The rate constant chosen is removed from the list and a new one is added.
    
    Parameters
    ----------
    rate_constant_list : list(dict)
        The list of rate constants to pick from with each element containing a 
        dictionairy describing the rate constant properties.
    rates : np.ndarray
        The numerical value of the rate constants.
    current_time : float
        The current time in the simulation in seconds.

    Returns
    ----------
    list(dict)
        A new list of the rate constants with any new rate constants with the 
        new step.
    np.ndarray
        The numerical value of the rate constants with the new rate const.
    """
    # first randomly pick a number
    return None