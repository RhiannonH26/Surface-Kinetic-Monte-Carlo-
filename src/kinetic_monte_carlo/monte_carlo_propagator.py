import numpy as np

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
    comps_properties : dict(dict)
        A dictionary of dictionaries containing the properties of each element 
        in each dictionary. It must contain the following properties for each 
        dictionary (specified by the element): "partial_pressure" in atm, 
        "mass" in u (atomic units), and "k_ads" the rate constant in units of s^-1. 
        Note that the chosen name or "key" for each component in the first level 
        of the dictionary is unimportant. For example, one element of this list 
        could be:
    
    ```
    "<atom>" : {   # <atom> is the name of the element, str
        "partial_pressure" : float, # partial pressure of the component
        "mass" : float, # mass in u (g/mol) of the component
        "k_ads" : float, # adsorption rate constant
        "k_des": float, # desorption rate constant
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
    for el in list(comps_properties.keys()):
        for l in range(N_grid):
            for w in range(N_grid):
                myk = comps_properties.get(el).get("k_ads")
                mydict = {"i": None,
                          "j": [l,w],
                          "i_atom": el}
                rate_constant_properties.append(mydict)
                rates[l,w] = myk
    # note could have stored things a bit differently but this is how i chose to do it
    return rate_constant_properties, rates.squeeze()

def _choose_random_rate(rates: np.ndarray):
    """Picks a random rate constant, k_q, using binary search
    according to the equation:

    .. math::

        \sum_{i=1}^{p} k_i \gt \rho_1 k_{tot} \gt \sum_{i=1}^{p-1} k_i 
    
    Parameters
    ----------
    rates: np.ndarray
        A numpy array of size (N,) containing all N possible rates the system
        has.
    
    Returns
    ----------
    int
        Index of the chosen rate
    """
    # pick a random number
    rho_1 = np.random.random()
    # get total rates
    cumsum = np.cumsum(rates)
    # start binary search, using numpy's search
    chosen = np.searchsorted(cumsum,rho_1*cumsum[-1]) 
    return chosen #### NEED TO ALSO REMOVE OTHER THINGS FOR SAME POSITION

def propagate_monte_carlo_one_step(rate_constant_list: list,
                                   rates: np.ndarray,
                                   current_time: float,
                                   comps_properties: dict,
                                   ):
    """Propagates the kinetic monte carlo algorithm in time by taking one step.
    
    The algorithm works by randomly choosing a rate constant weighted by the size of the
    rate constant and then propagating in time proportional to the inverse of the rate
    constant multiplied by a random number to obey the Poisson distribution (i.e. could
    leave faster than expected).

    The rate constant chosen is removed from the list and a new one is added. Done according
    to https://doi.org/10.3389/fchem.2019.00202.
    
    Parameters
    ----------
    rate_constant_list : list(dict)
        The list of rate constants to pick from with each element containing a 
        dictionairy describing the rate constant properties.
    rates : np.ndarray
        The numerical value of the rate constants.
    current_time : float
        The current time in the simulation in seconds.
    comps_properties : list(dict)
        A list of dictionaries containing the properties of each element 
        in each dictionary. It must contain the following properties for each 
        keyword: the element specified with "atom", "partial_pressure" in atm, 
        "mass" in u (atomic units), and "k_ads" the rate constant in units of s^-1. 
        Note that the chosen name or "key" for each component in the first level 
        of the dictionary is unimportant. For example, one element of this list 
        could be:

    Returns
    ----------
    list(dict)
        A new list of the rate constants with any new rate constants with the 
        new step.
    np.ndarray
        The numerical value of the rate constants with the new rate const.
    """
    # first randomly pick a number
    rho_2 = np.random.random() # rho_1 is in the random rate chooser
    # get k_tot
    k_tot = np.sum(rates)
    # draw a random process (index) using binary search
    index = _choose_random_rate(rates)
    # propagate according to the TOTAL rate constant to ensure that it occurs correctly
    t -= np.log(rho_2)/k_tot
    # update the rate lists by 
    # (1) removing the current rate
    rates = np.delete(rates, index) ### MIGHT BE INEFFICIENT, CONSIDER CHANGING DATA STRUCTURE
    curr_rate_info = rate_constant_list.pop(index)
    element = curr_rate_info.get('i_atom')
    # (2) adding new rates for current position
    # check to see what new rates are possible based on what rate was executed
    if curr_rate_info.get('i') == None: 
        # then we just executed an adsorption event
        # the only next possible event is a desorption event, add that
        kdes = comps_properties.get(element).get('k_des')
        location = curr_rate_info.get("j") # should be of size (2,)
        # add it to the things we care about
        rates.append(kdes)
        rate_constant_list.append({"i": None,
                          "j": location,
                          "i_atom": element})
    elif curr_rate_info.get('j') == None:
        # then we just executed a desorption event
        # the next possiblity for the SITE is an adsorption event
        # you could have adsorption of ANY of the molecules
        pass
    return None