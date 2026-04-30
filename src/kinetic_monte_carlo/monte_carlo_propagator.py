import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_full_ads_des_list(N_grid: float,
                               comps_properties: dict,
                               surface_smoothness: float,
                               k_indices: dict,
                               seed = None):
    """
    Generate rate constant list for a surface approximated by a 2D N_grid x N_grid
    surface containing the adsorption and desorption rate constants based on the
    surface_smoothness. If `surface_smoothness == 0.`, then all adsorption and desorption sites
    are assumed to be equivalent. Otherwise, a value from > 0 indicates how rough the surface is.
    Lower values are associated with a more rough surface, and higher values are associated with
    a less rough surface.

    Assumes that the surface starts out clean, i.e. no desorption events are possible.

    The output is a np.array that contains an array of the numerical rate
    constants that has an index that matches the description above. The index
    provides information about (1) the location on the grid (first two indices),
    (2) the atom type, and (3) the rate constant type (ads, des, ...)

    Parameters
    ----------
    N_grid : float
        The number of positions along one direction of the 2D metal surface.
    comps_properties : dict[dict]
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
    surface_smoothness : float
        A value of 0 indicates no roughness, a value > 0 indicates the degree of
        roughness on the surface. A very rough surface has smaller values and
        A high smoothness has larger values. In general higher values of smoothness 
        mean that
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
        Seed for the process, if provided
    
    Returns
    ----------
    np.ndarray(N_grid, N_grid, N_components, max_N_rxn_types_per_atom)
        The numerical value of the rate constants for both adsorption and
        desorption. (ASSUMES ONLY ADSORPTION DESORPTION ARE POSSIBLE)
    """
    if seed:
        np.random.seed(seed)
    rates = np.zeros((N_grid,N_grid,len(comps_properties),2))
    ads = k_indices.get("k_ads")
    des = k_indices.get("k_des")
    # do this for both, only make modifications if defects are present
    for k in range(len(comps_properties)): # for k in number of components
        el = list(comps_properties.keys())[k] # current element
        # now set both adsorption and desorption
        rates[:,:,k,ads] = comps_properties.get(el).get("k_ads") # set all grid elements to be k_ads
        rates[:,:,k,des] = comps_properties.get(el).get("k_des") # set all grid elements to be k_ads
    if surface_smoothness != 0:
        random_noise = np.random.random((N_grid,N_grid))
        Z = 4*gaussian_filter(random_noise,sigma=surface_smoothness)
        # get some filter to apply to rates
        # want higher ads and lower des to go together to simulate trapping
        # sites on a surface
        ads_array = Z/np.max(Z) * 8 - 6
        des_array = np.max(ads_array)-ads_array.copy()+np.min(ads_array) # get the inverse amount, minimum value is the same minimum value as ads
        # multiply all rate constants by this factor for all adsorbates
        rates[:,:,:,ads] *= np.repeat(ads_array[:,:,np.newaxis],len(comps_properties),axis=2)
        rates[:,:,:,des] *= np.repeat(des_array[:,:,np.newaxis],len(comps_properties),axis=2)
    return rates

def generate_rate_const_initial_list(full_rates: np.ndarray,
                                    k_indices: dict):
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
    full_rates : np.ndarray
        A list of numerical rates (adsorption and desorption) for the system, either
        made with the helper function `generate_full_ads_des_list` or generated
        by the user.
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
    
    Returns
    ----------
    np.ndarray(N_grid, N_grid, N_components, max_N_rxn_types_per_atom)
        The numerical value of the rate constants. 
    """
    # the positions provide information about what rate it is
    # the last number in the array tells us what type of rxn is occuring
    # an index of 0 indicates an adsorption rate constant
    # an index of 1 indicates a desorption rate constant
    # ... and so on in the order that they appear GIVEN that they start with k_...
    # at the beginning, only desorption events are possible
    initial_rates = full_rates.copy()
    initial_rates[:,:,:,k_indices.get("k_des")] = 0.
    # note could have stored things a bit differently but this is how i chose to do it
    return initial_rates

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
    chosen = min(chosen, rates.size - 1) # avoid bugs
    # fix the index
    real_index = np.unravel_index(chosen,rates.shape)
    return real_index

def propagate_monte_carlo_one_step(rates: np.ndarray,
                                   current_time: float,
                                   k_indices: dict,
                                   grid_vis: np.ndarray,
                                   tot_array: np.ndarray,
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
    grid_vis : np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (Ngrid, Ngrid)
    tot_rates_array : np.ndarray
        All possible adsorption and desorption events that could happen on the surface
        regardless of what events has occured. Same dimensions as `rates`. Serves as
        storage for the rates; used for pulling rate constant for the next step.
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
    np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (Ngrid, Ngrid)
    """
    if seed:
        np.random.seed(seed)
    tot_rates_array = tot_array.copy()
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
        rates[*index[:-1],k_indices.get("k_des")] = tot_rates_array[*index[:-1],k_indices.get("k_des")]
        # change the grid for this to be the atom identity = comps_properties + 1, 0 represents nothing there
        grid_vis[index[0], index[1]] = index[2] + 1
    elif event_type == k_indices.get('k_des'): # if we picked a desorption event
        # first remove the desorption
        rates[index] = 0.
        # change the grid for this to be the atom identity; 0 represents nothing there
        grid_vis[index[0], index[1]] = 0
        # allow ANY components to adsorb here
        # recall: 3rd index is the elements, just select all
        rates[*index[:2],:,k_indices.get('k_ads')] = tot_rates_array[*index[:2],:,k_indices.get('k_ads')]
    return rates,current_time,grid_vis

if __name__ == "__main__":
    N_grid = 100
    comps_properties = {
        "A" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 7e-1,
            "k_des" : 2.0e-1,
        },
        "B" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1.0e-1,
            "k_des" : 1.0e-9,
        }
    }
    surface_smoothness = 8.0
    generate_full_ads_des_list(N_grid,comps_properties,surface_smoothness)