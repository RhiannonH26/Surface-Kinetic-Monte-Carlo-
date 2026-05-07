## Surface Kinetic Monte Carlo
#### For Catalysis and Layer Deposition Applications

This Python package aims to simulate the time-dependent behaviour
of adsorption onto a simplified 2D metal surface. It is useful for applications
such as surface catalysis. Any number of components will work.

The current state of the software makes the following assumptions:
- The gas is sufficiently saturated such that adsorption is not time-dependent
- Lateral interactions between adsorbates are negligible
- Only adsorption events are possible

It is also currently only capable of running a simulation on a surface
where the active sites are equal (i.e. adsorption and desorption for
a given component are the same for all sites) *or* random roughness.

### Installation
To install this package, run the following in the cloned directory:

`pip install -e .`

### Running the Simulation
To run the package (i.e. to make plots), one can specify the parameters in
the `./src/kinetic_monte_carlo/run_kmc` file and run it using the following
run from the `./` directory.

```
python -m kinetic_monte_carlo.run_kmc
```

## Examples

#### Thermodynamic vs kinetic adsorbates

A concept many have seen in early Organic Chemistry courses are the existence
of thermodynamic and kinetic products. Thermodynamic products are generally more
stable but are generated very slowly compared to kinetic products, which are
generated fast but are short-lived due to the low Gibbs free energy. An example
of this in the context of adsorbates onto a (clean) metal surface is shown below.

```
comps_properties = {
        "A" : {
            "partial_pressure" : 0.7,
            "mass" : 15.999,
            "k_ads" : 7e-1,
            "k_des" : 7e-1,
        },
        "B" : {
            "partial_pressure" : 0.3,
            "mass" : 12.01,
            "k_ads" : 1.0e-1,
            "k_des" : 0.5e-1,
        }
    }
    max_time = 100 # seconds
    N_grid = 75
    num_steps = 100
    surface_smoothness="max"
    el_index = {1:["red","A"],2:["blue","B"]}

    print("Running monte carlo")
    full_rates,grid_vis_over_time,time_grid = run_full_length_monte_carlo(N_grid,comps_properties,max_time,surface_smoothness)

    print("Done. Plotting now.")
    t_uniform,grid_vis_uniform = map_to_time_const_grid(num_steps, max_time, grid_vis_over_time, time_grid)

    plot_trajectory(N_grid,grid_vis_uniform,el_index,t_uniform)
    plot_surface_coverage(N_grid,grid_vis_over_time,el_index,time_grid)
```

Which outputs the following plots:

Inline-style: 
![alt text]([https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png](https://github.com/RhiannonH26/Surface-Kinetic-Monte-Carlo-/blob/main/examples_plots/surface_coverage_defects.png) "Logo Title Text 1")

### Rough Surface Adsorption

We know that when the surface is not perfect, due to defects present on the surface, the
rates of adsorption and desorption depend on the site. In particular, some sites act as
"traps" that increase the rate of adsorption and decrease the rate of desorption. We can 
visualize this effect by creating a random surface with random "trap" sites. This code
will show the standard plots used in the above example while also creating a plot to show
the surface roughness. Lighter colours indicate a higher rate of adsorption and lower rate
of desorption (i.e. trap sites) and darker colours indicate the opposite. 

```
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
```

Which outputs the following plots:
