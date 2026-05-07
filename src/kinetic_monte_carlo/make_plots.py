import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib
from kinetic_monte_carlo.monte_carlo_propagator import generate_full_ads_des_list
import numpy as np

def make_grid(ax : plt.Axes, N_grid : int,grid_vis : np.ndarray,el_index : dict,curr_time: float):
    """
    Plots the current state of the adsorbed surface. Each site
    represents a circle. An unfilled circle represents an unoccupied site
    and a filled circle represents an occupied site. Time is indicated 
    at the top of the plot. This is a helper function to `plot_trajectory`.

    Parameters
    ----------
    ax : plt.Axes
        An Axes object which represent a snapshot of the current state of the system.
        i.e. filled/unfilled circles and the timestep.
    N_grid : int
        Number of grid points.
    grid_vis : np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (Ngrid, Ngrid)
    el_index : dict[list]
        Indicates the index of the atom in the original `comps_properties` plus 1
        (0 is reserved for an unfilled circle) as well as the colour associated with
        the molecule and the molecule name in a list. The general format is:
            ```
            el_index = {
                1 : ['red','A'],
                2 : ['blue','B']
                ...
            }
            ``` 
    curr_time : float
        The current time of the system.
    
    Returns
    ----------
    plt.Axes.artists
        Returns the artists for plotting a trajectory (i.e. the filled and unfilled circles
        in a certain frame.)
    """
    ax.clear() # clear prev. frame
    for i in range(N_grid):
        for j in range(N_grid):
            if grid_vis[i,j] == 0:
                circle = plt.Circle((i,j),0.45,color="grey",fill=False,linewidth=0.2)
            else:
                circle = plt.Circle((i,j),0.45,color=el_index.get(int(grid_vis[i,j]))[0],fill=True,linewidth=0.2)
            ax.add_artist(circle)
    plt.xlim([-1.0, N_grid])
    plt.ylim([-1.0, N_grid])
    ax.set_aspect(1.0)
    plt.scatter(np.arange(N_grid),np.arange(N_grid),s=0, facecolors='none')
    plt.axis('off')
    plt.title(f"t = {curr_time:.1f} s")
    # make legend
    patches = []
    for elle in el_index:
        patches.append(mpatches.Patch(color=el_index.get(elle)[0], label=el_index.get(elle)[1]))

    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.show() ### SHOWS A PLOT
    return ax.artists ### helped by ChatGPT and Stack Overflow

def plot_trajectory(N_grid: int,grid_vis: dict,el_index: dict,t_uniform:np.ndarray,output_filename = "trajectory.mp4"):
    """Plots the trajectory as a function of time resulting from the kinetic
    monte carlo simulation. Shows the coverage of the system visually by depicting 
    the molecules as filled circles and empty active sites as unfilled circles. 

    Parameters
    ----------
    N_grid : int
        Number of grid points.
    grid_vis : np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (Ngrid, Ngrid)
    el_index : dict[list]
        Indicates the index of the atom in the original `comps_properties` plus 1
        (0 is reserved for an unfilled circle) as well as the colour associated with
        the molecule and the molecule name in a list. The general format is:
            ```
            el_index = {
                1 : ['red','A'],
                2 : ['blue','B']
                ...
            }
            ``` 
    t_uniform : np.ndarray
        Uniformly spaced time points. Calculated using the helper function in 
        `run_kmc.map_to_time_const_grid`.
    output_filename : str
        Name of the output file for the video of the trajectory.

    Returns
    ----------
    None
        Outputs the video to the current directory.
    """
    fig, ax = plt.subplots(dpi=400)

    def update(frame):
        return make_grid(ax, N_grid, grid_vis[frame], el_index, t_uniform[frame])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(grid_vis),
        interval=500,
        blit=True,
        repeat=False
    )

    ani.save(output_filename, writer="ffmpeg", fps=5)
    plt.show()

def plot_surface_coverage(N_grid: int,grid_vis: dict,el_index: dict,times: np.ndarray,output_filename = "surface_coverage.png"):
    """Plots the surface coverage as a function of time on a plot.
    Plots a line for each of the components.

    Parameters
    ----------
    N_grid : int
        Number of grid points.
    grid_vis : np.ndarray
        Provides information about the state of the grid for later visualization. Each 
        position has an index indicating which atom is adsorbed (equal to 1 + index
        in the comps_properties dict). Size (Ngrid, Ngrid)
    el_index : dict[list]
        Indicates the index of the atom in the original `comps_properties` plus 1
        (0 is reserved for an unfilled circle) as well as the colour associated with
        the molecule and the molecule name in a list. The general format is:
            ```
            el_index = {
                1 : ['red','A'],
                2 : ['blue','B']
                ...
            }
            ``` 
    time : np.ndarray
        Non-uniform time steps (directly from the kinetic Monte Carlo algorithm).
    output_filename : str
        Name of the output file for the video of the trajectory.

    Returns
    ----------
    None
        Outputs the plot to the current directory.
    """
    # first get the surface coverage for each element
    plt.figure(dpi=400)
    nums = {l:[] for l in el_index}
    l_max = np.max(list(el_index.keys()))
    for i in range(len(grid_vis)):
        for l in range(1,l_max+1):
            nums[l].append(np.count_nonzero(grid_vis[i] == l)/N_grid**2) # this represents the surface covg
    # now plot it
    for l in range(1,l_max+1):
        plt.plot(times,nums.get(l),color=el_index.get(l)[0],label=el_index.get(l)[1])
    plt.xlabel('Time (s)')
    plt.ylabel('Surface coverage, theta')
    plt.legend()
    plt.savefig(output_filename)
    plt.show()

def plot_contour(N_grid: int,tot_rates : np.ndarray):
    """
    Plots the magnitude of the adsorption rate constant at each grid point
    specified in the `tot_rates` object.

    Pick colours based on magnitude of rate constants and normalizes them
    for ease of plotting.
    
    Parameters
    ----------
    N_grid : int
        Number of grid points.
    tot_rates : np.ndarray
        The rates at each position for adsorption and desorption and for each component.
        Of size (N_grid, N_grid, N_components, N_events). Here we assume only adsorption
        and desorption events are possible i.e; N_events=2.

    Returns
    ----------
    None
        Outputs the plot to the current directory.
    """
    fig,ax = plt.subplots(dpi=400)

    # normalize things (note i did not fully make this myself, ChatGPt helped with syntax)
    # for tot_rates[i,j,0,0], doesn't matter that I picked index 0, since same filter is applied to both
    # when the tot_rates list was created
    values = tot_rates[:,:,0,0]
    norm = matplotlib.colors.Normalize(vmin=np.min(values),vmax=np.max(values))
    cmap = matplotlib.cm.plasma
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])

    for i in range(N_grid):
        for j in range(N_grid):
            # also we are just looking at ads
            color = cmap(norm(tot_rates[i,j,0,0]))
            circle = plt.Circle((i,j),0.45,color=color,fill=True,linewidth=0.2)
            ax.add_artist(circle)
    ax.set_xlim([-1.0, N_grid])
    ax.set_ylim([-1.0, N_grid])
    ax.set_aspect(1.0)
    plt.scatter(np.arange(N_grid),np.arange(N_grid),s=0, facecolors='none')
    ax.axis('off')
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_ticks([])
    plt.savefig("surface_defects.png")
    plt.show() ### SHOWS A PLOT

if __name__ == "__main__":
    # get el_index from comps_properties
        # checks that desorption events are handled correctly
    np.random.seed(67)
    comps_properties = {
        "O" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 1.0,
            "k_des" : 2.0,
        },
        "C" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 12.0,
            "k_des" : 4.0,
        }
    }
    # mostly zeros, 2x2 grid with 2 components, 2 rxn types
    k_indices = {'k_ads': 0, 'k_des': 1}
    # only one site is active
    surface_smoothness=5.0
    tot_rates = generate_full_ads_des_list(30,comps_properties,surface_smoothness,k_indices,seed=67)
    plot_contour(30,tot_rates)