import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
                circle = plt.Circle((i,j),0.45,color=el_index.get(int(grid_vis[i,j])),fill=True,linewidth=0.2)
            ax.add_artist(circle)
    plt.xlim([-1.0, N_grid])
    plt.ylim([-1.0, N_grid])
    ax.set_aspect(1.0)
    plt.scatter(np.arange(N_grid),np.arange(N_grid),s=0, facecolors='none')
    plt.axis('off')
    plt.title(f"t = {curr_time:.1f} s")
    #plt.show() ### SHOWS A PLOT
    return ax.artists 

def plot_trajectory(N_grid,grid_vis,el_index,t_uniform):
    """Plots the trajectory as a function of time.

    Each "row" of grid_vis represents a time_step of N_timesteps.

    The kMC must be preprocessed before this to ensure that it is actually
    on a grid of time.
    """
    fig, ax = plt.subplots()

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

    ani.save("trajectory.mp4", writer="ffmpeg", fps=20)
    plt.show()
    

if __name__ == "__main__":
    # get el_index from comps_properties
    el_index = {1:"red",2:"blue"}
    grid_vis = np.random.randint(3,size=(4,50,50))
    t_uniform = np.linspace(0,10,4)

    plot_trajectory(50,grid_vis,el_index,t_uniform)