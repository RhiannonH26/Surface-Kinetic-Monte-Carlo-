import matplotlib.pyplot as plt
import numpy as np
import argparse

def make_grid(N_grid,grid_vis,el_index):
    """
    Plots the current state of the adsorbed surface. Each site
    represents a circle. An unfilled circle represents an unoccupied site
    and a filled circle represents an occupied site.

    Pick the colours for the filled circles in el_index. grid_vis is passed from another function

    FIX DOCTSRINGS!!!!!!!
    #####################
    #####################

    """
    plt.figure()
    ax = plt.gca()
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
    plt.scatter(np.arange(N_grid),np.arange(N_grid),s=0, cmap='jet', facecolors='none')
    plt.axis('off')
    plt.show()
    plt.savefig("result")

if __name__ == "__main__":
    # get el_index from comps_properties
    el_index = {1:"red",2:"blue"}
    grid_vis = np.random.randint(3,size=(50,50))

    make_grid(50,grid_vis,el_index)