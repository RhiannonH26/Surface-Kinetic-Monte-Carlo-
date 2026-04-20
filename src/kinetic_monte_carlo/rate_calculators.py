import numpy as np

# k=8.617333262e-5 #eV⋅K−1

def k_adsorption(p_A: float,
               k: float,
               T: float,
               m_A: float):
    """Calculates the adsorption rate constant for a given species A.
    Assumes sticking probability is one.

    Parameters
    ----------
    p_A : float
        The partial pressure of species `A` in units of Pa
    k : float
        Boltzmann constant in units of J K^-1
    T : float
        The temperature in units of K
    m_A : float
        The mass of species A in units of kg
    
    Returns
    ----------
    float
        Adsorption rate constant for species A.
    """
    return p_A/np.sqrt(2*np.pi*m_A*k*T)

def k_desorption(delG_ads: float,
                 p_A: float,
                 k: float,
                 T: float,
                 m_A: float):
    """Calculates the desorption rate constant for a given species A.
    Assumes sticking probability is one.

    Parameters
    ----------
    delG_ads : float
        The delta Gibbs free energy of adsorption in **units of eV**.
    p_A : float
        The partial pressure of species `A` in units of Pa
    k : float
        Boltzmann constant in units of J K^-1
    T : float
        The temperature in units of K
    m_A : float
        The mass of species A in units of kg
    
    Returns
    ----------
    float
        Adsorption rate constant for species A.
    """
    # first convert to units of joules
    ev_to_kcal = 1.60218e-19
    delG = ev_to_kcal*delG_ads
    # now get k_des according to detailed balance
    k_ads = k_adsorption(p_A,k,T,m_A)
    k_des = k_ads * np.exp(delG/(k*T))
    return k_des

