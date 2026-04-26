import numpy as np

def _count_number_rxn_types(comps_properties: dict):
    """Determines the number of reaction types based on the number of
    k_* present in the comps_properties dictionary (which also has
    nested dictionaries for each component). 
    
    Parameters
    ----------
    comps_properties : dict(dict)
        A dictionary of dictionaries containing the properties of each element 
        in each dictionary. It must contain the following properties for each 
        dictionary (specified by the element): "partial_pressure" in atm, 
        "mass" in u (atomic units), and "k_ads" the rate constant in units of s^-1. 
        
        Note that the rate constants must be in the same order for ALL components.
        
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
    int
        The maximum number of reaction types for all components for
        matrix reconstruction.
    """
    total_rxn_types = np.max([len([h for h in comps_properties.get(el).keys() if h.startswith("k_")]) for el in comps_properties])
    return total_rxn_types

def generate_rate_const_initial_list(N_grid: float,
                                    comps_properties: dict):
    """Calculates the initial rate constant list for a surface approximated by
    a 2D N_grid x N_grid surface. All surface points are assumed to be the same
    and are a location for a particle to adsorb. This assumes that the surface 
    starts out clean, i.e. no desorption events are possible.

    The output is a np.array that contains an array of the numerical rate
    constants that has an index that matches the description above. The index
    provides information about (1) the location on the grid (first two indices),
    (2) the atom type, and (3) the rate constant type (ads, des, ...)

    Parameters
    ----------
    N_grid : int
        Number of surface locations along one dimension of the surface.
    comps_properties : dict(dict)
        A dictionary of dictionaries containing the properties of each element 
        in each dictionary. It must contain the following properties for each 
        dictionary (specified by the element): "partial_pressure" in atm, 
        "mass" in u (atomic units), and "k_ads" the rate constant in units of s^-1. 
        
        Note that the rate constants must be in the same order for ALL components.
        Can have different lengths but the indices of the common components
        should be the same.
        
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
    np.ndarray(N_grid, N_grid, N_components, max_N_rxn_types_per_atom)
        The numerical value of the rate constants. 
    """
    num_rxn_types = _count_number_rxn_types(comps_properties)
    rates = np.zeros((N_grid,N_grid,len(comps_properties.keys()),num_rxn_types))
    # the positions provide information about what rate it is
    # the last number in the array tells us what type of rxn is occuring
    # an index of 0 indicates an adsorption rate constant
    # an index of 1 indicates a desorption rate constant
    # ... and so on in the order that they appear GIVEN that they start with k_...
    # note that this does NOT work well if there are defects and position matters
    for k in range(len(comps_properties.keys())):
        for l in range(N_grid):
            for w in range(N_grid):
                el = list(comps_properties.keys())[k]
                myk = comps_properties.get(el).get("k_ads")
                rates[l,w,k,0] = myk
    # note could have stored things a bit differently but this is how i chose to do it
    return rates

def _choose_random_rate(rates: np.ndarray, seed = None):
    """Picks a random rate constant, k_q, using binary search
    according to the equation:

    .. math::

        \sum_{i=1}^{p} k_i \gt \rho_1 k_{tot} \gt \sum_{i=1}^{p-1} k_i 
    
    Parameters
    ----------
    rates : np.ndarray
        A numpy array of size (N_grid,N_grid,N_components,N_max_rates) containing all N possible rates the system
        has.
    seed : int or None
        Seed of random number, if desired.
    
    Returns
    ----------
    np.ndarray
        Index of the chosen rate
    """
    if seed:
        np.random.seed(seed)
    # pick a random number
    rho_1 = np.random.random()
    # get total rates
    cumsum = np.cumsum(rates.flatten())
    # start binary search, using numpy's search
    chosen = np.searchsorted(cumsum,rho_1*cumsum[-1]) 
    # fix the index
    real_index = np.unravel_index(chosen,rates.shape)
    return real_index

def propagate_monte_carlo_one_step(rates: np.ndarray,
                                   current_time: float,
                                   comps_properties: dict,
                                   k_indices: dict,
                                   seed = None,
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
    rates : np.ndarray
        The numerical value of the rate constants with their positions providing info
        about the type of event.
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
    k_indices : dict
        Contains the indices (in the 4th dimension of `rates`) of each rate constant.
        Should have the following key words at minimum:
            ```
            k_indices = {
                'k_ads': 0,
                'k_des': 1,
                ...
            }
            ```
        Note that this is not generalizable and requires more effort in future versions.
    seed : int or None
        Seed of random choice, if chosen.

    Returns
    ----------
    np.ndarray
        The numerical value of the rate constants with the new rate const and removed rate constants.
    float
        Updated time.
    """
    if seed:
        np.random.seed(seed)
    # first randomly pick a number
    rho_2 = np.random.random() # rho_1 is in the random rate chooser
    # get k_tot
    k_tot = np.sum(rates)
    # draw a random process (index) using binary search
    index = _choose_random_rate(rates,seed=seed)
    # propagate according to the TOTAL rate constant to ensure that it occurs correctly
    current_time -= np.log(rho_2)/k_tot
    # update the rate lists by 
    # (1) removing the current rate AND the possibility of any other adsorption events at this site
    # (2) adding new rates for current position
    # to do this, we first need to check what event just happened
    event_type = index[-1]
    if event_type == k_indices.get('k_ads'): # if we just picked an adsorption event
        # prevent any OTHER components from adsorbing at this site
        rates[index[0],index[1],:,event_type] = 0.
        # set the desorption for this atom
        el = list(comps_properties.keys())[index[2]] # get the atom from chosen index index
        rates[index[0],index[1],index[2],k_indices.get("k_des")] = comps_properties.get(el).get("k_des")
    elif event_type == k_indices.get('k_des'): # if we picked a desorption event
        # first remove the desorption
        rates[index] = 0.
        # allow ANY components to adsorb here
        # go element by element here
        for m in range(len(comps_properties.keys())):
            el = list(comps_properties.keys())[m]
            myk = comps_properties.get(el).get("k_ads")
            rates[index[0],index[1],m,k_indices.get('k_ads')] = myk
    return rates,current_time