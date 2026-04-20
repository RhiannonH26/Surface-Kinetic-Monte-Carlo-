import numpy as np
from typing import Callable

def generate_rate_const_initial_list(N_grid: float,
                                    k_ads: Callable,
                                    k_des: Callable,
                                    **kwargs):
    """Calculates the initial rate constant list for a surface approximated by
    a 2D N_grid x N_grid surface. All surface points are assumed to be the same
    and are a location for a particle to adsorb. We will assume that only
    adsorption and desorption events are possible.

    The list is contains a series of dictionaries specifying the following:
    ```
    {   kij : float, # rate constant
        i : list(int) or None, # None if exists as a gas in state i
        j : list(int) or None, # None if exists as a gas in state j
        i_atom: str, # identity of atom in state i
    }
    ```

    Parameters
    ----------
    N_grid : int
        Number of surface locations along one dimension of the surface.
    k_ads : Callable
        The function that calculates the adsorption rate constant.
    k_des : Callable
        The function that calculates the desorption rate constant.
    
    Returns
    ----------
    list(dict)
        The list of rate constants to pick from.
    """
    return None

def generate_neighbour_list(N_grid: int):
    return None