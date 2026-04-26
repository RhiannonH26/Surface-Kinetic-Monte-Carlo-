import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinetic_monte_carlo.make_plots import plot_trajectory
from kinetic_monte_carlo.monte_carlo_propagator import propagate_monte_carlo_one_step,generate_rate_const_initial_list

def generate_k_indices(comps_properties):
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

def map_to_time_const_grid(num_steps, max_time, grid_vis_over_time, time_grid):
    # first get the new time grid
    equal_time_grid = np.linspace(0,max_time,num_steps)

    # find which original interval each uniform time falls into
    indices = np.searchsorted(time_grid, equal_time_grid, side='right') - 1

    # clamp to valid range (prevent indices <0 or > -1)
    indices = np.clip(indices, 0, len(time_grid) - 1)
    print(indices)
    print(grid_vis_over_time[0])

    return equal_time_grid,np.array(grid_vis_over_time)[indices] # now projected onto a uniform grid

def run_full_length_monte_carlo(N_grid,comps_properties,max_time):

    rates = generate_rate_const_initial_list(N_grid, comps_properties)
    grid_vis = np.zeros((N_grid,N_grid))
    current_time = 0
    k_indices = generate_k_indices(comps_properties)

    ### for storing
    grid_vis_over_time = [grid_vis]
    print(grid_vis_over_time)
    time_grid = [current_time]
    ####

    while current_time < max_time:
        # run a step
        rates,current_time,grid_vis = propagate_monte_carlo_one_step(rates,current_time,comps_properties,k_indices,grid_vis)
        # save things into the list
        grid_vis_over_time.append(grid_vis.copy())
        time_grid.append(current_time)
    
    return grid_vis_over_time,time_grid

def main():
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 5e-1,
            "k_des" : 2.0e-1,
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1.0e-1,
            "k_des" : 4.0e-2,
        }
    }
    max_time = 10 # seconds
    N_grid = 20
    num_steps = 10
    el_index = {1: 'red', 2: 'blue'}

    print("Running monte carlo")
    grid_vis_over_time,time_grid = run_full_length_monte_carlo(N_grid,comps_properties,max_time)

    print("Done. Plotting now.")
    t_uniform,grid_vis_uniform = map_to_time_const_grid(num_steps, max_time, grid_vis_over_time, time_grid)

    plot_trajectory(N_grid,grid_vis_uniform,el_index,t_uniform)

if __name__ == "__main__":
    main()

