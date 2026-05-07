import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinetic_monte_carlo.make_plots import plot_trajectory, plot_surface_coverage, plot_contour
from kinetic_monte_carlo.monte_carlo_propagator import propagate_monte_carlo_one_step,generate_rate_const_initial_list,generate_full_ads_des_list

def generate_k_indices(comps_properties: dict):
    """
    Generates the indices for each rate constant in comps_properties
    in the form of a dictionary.

    Parameters
    ----------
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

    Returns
    ----------
    dict
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
    """
    # generate k_indices
    # only use the first element of the comps_properties list
    # assume that both components have the same order (the user was already warned about this)
    key = list(comps_properties.keys())
    # find the index of the key with the longest attributes (i.e. longest num of k)
    pt = np.argmax([len(comps_properties.get(key[i])) for i in range(len(key))])
    mykey = key[pt]
    i = 0
    k_indices = {}
    for el in list(comps_properties.get(mykey).keys()):
        if el.startswith("k_"):
            k_indices[el] = i
            i+=1
    return k_indices

def map_to_time_const_grid(num_steps:int, 
                           max_time:float,
                           grid_vis_over_time: np.ndarray,
                           time_grid: np.ndarray):
    """Converts the time steps returned by the kinetic Monte Carlo algorithm
    (which are non-uniformly spaced; not ideal for the video plotting) into
    uniformly placed timesteps using binary sort for all of the indices in the
    non-uniform time array to map it onto a uniform array.

    Parameters
    ----------
    num_steps : int
        Number of uniformly-spaced time points.
    max_time : float
        The max time set by the user in the original kinetic monte carlo code.
    grid_vis_over_time : np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (N_steps, Ngrid, Ngrid). Each "row" resembles
        the typical `grid_vis` array.
    time_grid : np.ndarray
        The non-uniform time steps taken by the kinetic Monte Carlo algorithm.

    Returns
    ----------
    np.ndarray
        Uniformly spaced time grid.
    """
    # first get the new time grid
    equal_time_grid = np.linspace(0,max_time,num_steps)

    # find which original interval each uniform time falls into
    indices = np.searchsorted(time_grid, equal_time_grid, side='right') - 1

    # clamp to valid range (prevent indices <0 or > -1)
    indices = np.clip(indices, 0, len(time_grid) - 1)

    return equal_time_grid,np.array(grid_vis_over_time)[indices] # now projected onto a uniform grid

def run_full_length_monte_carlo(N_grid:int,
                                comps_properties:dict,
                                max_time:float,
                                surface_smoothness:float):
    """
    Runs the full-length kinetic Monte Carlo simulation on a 2D metal
    surface for all of the components and rates specified in `comps_properties`.
    Rates are assumed to be either (a) the same regardless of the position (i.e.
    every site has the same adsorption/desorption rate for each component) or (b)
    a random smoothness given by the surface_smoothness variable. In principle, this
    function could be adjusted to allow for the user to provide their own site-specific
    list of rates. But this Python package is treated more as a demo package for
    teaching purposes.

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
    tuple(np.ndarray, np.ndarray, np.ndarray)
        At the first position is the numerical value of the rate constants 
        at all time steps (i.e. rateconstants array plus an added dimension for time steps)

        The second position contains information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (N_steps,Ngrid, Ngrid) (i.e. grid_vis
        plus a dimension for each time step).

        The third position indicates the time associated with each time step.
    """

    k_indices = generate_k_indices(comps_properties)
    full_rates = generate_full_ads_des_list(N_grid,comps_properties,surface_smoothness,k_indices)
    rates = generate_rate_const_initial_list(full_rates,k_indices)
    grid_vis = np.zeros((N_grid,N_grid))
    current_time = 0

    ### for storing
    grid_vis_over_time = [grid_vis.copy()]
    time_grid = [current_time]
    ####

    while current_time < max_time:
        # run a step
        rates,current_time,grid_vis = propagate_monte_carlo_one_step(rates,current_time,k_indices,grid_vis,full_rates)
        # save things into the list
        grid_vis_over_time.append(grid_vis.copy())
        time_grid.append(current_time)
    
    return full_rates,grid_vis_over_time,time_grid

def main():
    """
    An example of a monte carlo simulation for running.
    """
    comps_properties = {
        "A" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 7e-2,
            "k_des" : 2e-2,
        },
        "B" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1e-2,
            "k_des" : 1e-2,
        }
    }
    max_time = 170 # seconds
    N_grid = 75
    num_steps = 100
    surface_smoothness=3.0
    el_index = {1:["red","A"],2:["blue","B"]}

    print("Running monte carlo")
    full_rates,grid_vis_over_time,time_grid = run_full_length_monte_carlo(N_grid,comps_properties,max_time,surface_smoothness)

    print("Done. Plotting now.")
    t_uniform,grid_vis_uniform = map_to_time_const_grid(num_steps, max_time, grid_vis_over_time, time_grid)

    plot_trajectory(N_grid,grid_vis_uniform,el_index,t_uniform)
    plot_surface_coverage(N_grid,grid_vis_over_time,el_index,time_grid)
    plot_contour(N_grid,full_rates)

if __name__ == "__main__":
    main()

