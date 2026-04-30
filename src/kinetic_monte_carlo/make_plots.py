import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib
from kinetic_monte_carlo.monte_carlo_propagator import generate_full_ads_des_list
import numpy as np

def make_grid(ax,N_grid,grid_vis,el_index,curr_time):
    """
    Plots the current state of the adsorbed surface. Each site
    represents a circle. An unfilled circle represents an unoccupied site
    and a filled circle represents an occupied site.

    Pick the colours for the filled circles in el_index. grid_vis is passed from another function

    FIX DOCTSRINGS!!!!!!!
    #####################
    #####################

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
    return ax.artists 

def plot_trajectory(N_grid,grid_vis,el_index,t_uniform):
    """Plots the trajectory as a function of time.

    Each "row" of grid_vis represents a time_step of N_timesteps.

    The kMC must be preprocessed before this to ensure that it is actually
    on a grid of time.
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

    ani.save("trajectory.mp4", writer="ffmpeg", fps=5)
    plt.show()

def plot_surface_coverage(N_grid,grid_vis,el_index,times):
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
    plt.savefig("surface_coverage.png")
    plt.show()

def plot_contour(N_grid,tot_rates):
    """
    Plots the magnitude of the adsorption rate constant at each grid point.

    Pick colours based on magnitude of rate constants

    FIX DOCTSRINGS!!!!!!!
    #####################
    #####################

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
    return ax.artists 

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