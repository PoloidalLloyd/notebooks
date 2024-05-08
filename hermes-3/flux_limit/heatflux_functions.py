from scipy.integrate import simps
import matplotlib.pyplot as pltimport
from scipy.integrate import cumtrapz
import xhermes as xh
from boutdata.data import BoutData
from boutdata import collect
import matplotlib.pyplot as plt
import glob     
import re
import numpy as np
import pandas as pd

def spitzer_q_electron(dataframe):

    # Constants
    e = 1.602e-19  # Electron charge in Coulombs
    m_e = 9.109e-31  # Electron mass in kg
    pi = np.pi
    k0 = 13.58  # Given constant
    epsilon_0 = 8.85e-12  # Permittivity of free space in F/m

    # Assumed given values (you'll need to replace these with actual values)
    Z = 1  # Average ion charge, example value
    x = dataframe['y']
    Te = dataframe['Te']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']

    Y = 4 * pi * (e**2 / (4 * pi * epsilon_0 * m_e))**2

    ln_alpha = 6.6 - 0.5 * np.log(Ne/1e20) + 1.5* np.log(Te)

    v_t = np.sqrt(2 * e * Te/m_e)

    lambda_ei = (v_t**4)/(Y * Ni * ln_alpha)

    tau_t = lambda_ei/v_t

    grad_T = np.gradient(Te, x)

    # Unsure about the constants in this equation ((-1 +0.24)/(-1 + 4.2)) should it be + or - 1 for the electron
    q = -((Ne * e * Te)/(m_e)) * ((3 * np.sqrt(pi))/4) * (tau_t*k0) * ((1 +0.24)/(1 + 4.2)) *  grad_T

    # Convert from ev/m^2/s to W/m^2
    q_SH_electron = q * e

    return q_SH_electron

# def test_spitzer_q_ion(dataframe):

#     # Constants
#     e = 1.602e-19  # Electron charge in Coulombs
#     m_e = 9.109e-31  # Electron mass in kg
#     m_i = 2 * 1.67e-27  # Ion mass in kg
#     pi = np.pi
#     k0 = 13.58  # Given constant
#     epsilon_0 = 8.85e-12  # Permittivity of free space in F/m

#     # Assumed given values (you'll need to replace these with actual values)
#     n_e = 1e20  # Electron density in m^-3, example value
#     Z = 1  # Average ion charge, example value

#     x = dataframe['y']
#     Te = dataframe['Te']
#     Ti = dataframe['Td+']
#     Ne = dataframe['Ne']
#     Ni = dataframe['Nd+']
#     kappa_i = dataframe['kappa_par_d+']
#     # Y = 4*pi * ((e**2 )/ (4*pi*epsilon_0))**2

#     Y = 4 * pi * (e**2 / (4 * pi * epsilon_0 * m_e))**2
#     # print('Y',Y)s

#     ln_alpha = 6.6 - 0.5 * np.log(Ne/1e20) + 1.5* np.log(Te)
#     # print(ln_alpha)
#     # print('ln_alpha',ln_alpha)

#     v_t = np.sqrt(2 * e * Te/m_e)
#     # print('v_t', v_t)

#     lambda_ei = (v_t**4)/(Y * Ni * ln_alpha) 
#     print('lambda_ei_electron', lambda_ei)

#     tau_t = lambda_ei/v_t
#     # print('tau_t', tau_t)

#     grad_T = np.gradient(Ti, x)
#     # print('grad_T', grad_T)

#     q_SH = -((Ni * e * Ti)/(m_i)) * ((3 * np.sqrt(pi))/4) * (tau_t*k0) * ((1 +0.24)/(1 + 4.2)) *  grad_T

#     return q_SH

def spitzer_q_ion(dataframe, ion_mass_amu=2):

    # Constants
    e = 1.602e-19  # Electron charge in Coulombs
    m_e = 9.109e-31  # Electron mass in kg
    m_i = ion_mass_amu * 1.67e-27  # Ion mass in kg
    pi = np.pi
    k0 = 13.58  # Given constant
    epsilon_0 = 8.85e-12  # Permittivity of free space in F/m

    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']

    Y = 4 * pi * (e**2 / (4 * pi * epsilon_0 * m_i))**2

    ln_alpha = 6.6 - 0.5 * np.log(Ni / 1e20) + 1.5 * np.log(Ti)

    v_t_snb = np.sqrt(2 * e * Ti/m_i)

    lambda_ei_snb = (v_t_snb**4)/(Y * Ni * ln_alpha)
    print('lambda_ei_ion', lambda_ei_snb)

    tau_t_snb = (lambda_ei_snb)/(v_t_snb)

    grad_T_snb = np.gradient(Ti, x)

    q_SNB_ion = -((Ni * e * Ti)/(m_i)) * ((3 * np.sqrt(pi))/4) * (tau_t_snb*k0) * ((1 +0.24)/(1 + 4.2)) *  grad_T_snb

    # q is clculated in in ev/m^2/s so multiplying by e to get watts/m^2
    q_SNB_ion = q_SNB_ion * e
    return q_SNB_ion

def test_spitzer_q_ion(dataframe, ion_mass_amu=2):

    # Constants
    e = 1.602e-19  # Electron charge in Coulombs
    m_e = 9.109e-31  # Electron mass in kg
    m_i = ion_mass_amu * 1.67e-27  # Ion mass in kg
    pi = np.pi
    k0 = 13.58  # Given constant
    epsilon_0 = 8.85e-12  # Permittivity of free space in F/m

    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']
    kappa_i = dataframe['kappa_par_d+']

    Y = 4 * pi * (e**2 / (4 * pi * epsilon_0 * m_i))**2

    ln_alpha = 6.6 - 0.5 * np.log(Ni / 1e20) + 1.5 * np.log(Ti)

    v_t_snb = np.sqrt(2 * e * Ti/m_i)

    lambda_ei_snb = (v_t_snb**4)/(Y * Ni * ln_alpha)
    print('lambda_ei_ion', lambda_ei_snb)

    tau_t_snb = (lambda_ei_snb)/(v_t_snb)

    grad_T_snb = np.gradient(Ti, x)

    q_SNB_ion = -((Ni * e * Ti)/(m_i)) * ((3 * np.sqrt(pi))/4) * (tau_t_snb*kappa_i) * ((1 +0.24)/(1 + 4.2)) *  grad_T_snb

    # q is clculated in in ev/m^2/s so multiplying by e to get watts/m^2
    q_SNB_ion = q_SNB_ion * e
    return q_SNB_ion

def spitzer_q_electron_simple(dataframe):
    
    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']
    kappa_e = dataframe['kappa_par_e']
    kappa_i = dataframe['kappa_par_d+']

    # print(kappa_e)

    grad_T = np.gradient(Te, x)
    q = -kappa_e * grad_T

    return q

def spitzer_q_ion_simple(dataframe):
        
    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']
    kappa_e = dataframe['kappa_par_e']
    kappa_i = dataframe['kappa_par_d+']

    grad_T = np.gradient(Ti, x)

    q = -kappa_i * grad_T

    return q

def q_convective_electron(dataframe):
    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']
    Vi = dataframe['Vd+']
    Ve = dataframe['Ve']

    # Constants
    k_B = 1.38e-23  # Boltzmann constant, J/K

    # Calculate enthalpy per particle
    h = (5/2) * k_B * Te  # Ignoring binding energy for simplicity

    # Convective heat flux
    q_conv = Ne * h * Ve  # W/m^2 assuming area of flow is 1 m^2

    # print(f"Convective Heat Flux: {q_conv} W/m^2")


    return q_conv



def q_convective_ion(dataframe):
    """
    Calculate the convective heat flux for ions based on ion temperature, density, and bulk velocity.

    Parameters:
    - dataframe (pandas.DataFrame): A DataFrame containing the plasma parameters.

    Returns:
    - q_conv (numpy.ndarray): An array of the convective heat flux values in W/m^2.
    """
    # Ensure all necessary columns are in the dataframe
    required_columns = {'y', 'Te', 'Td+', 'Ne', 'Nd+', 'Vd+', 'Ve'}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Extract data from the dataframe
    x = dataframe['y']
    Ti = dataframe['Td+']
    Ni = dataframe['Nd+']
    Vi = dataframe['Vd+']

    # Constants
    k_B = 1.38e-23  # Boltzmann constant, J/K

    # Calculate enthalpy per particle
    h = (5/2) * k_B * Ti  # Ignoring binding energy for simplicity

    # Convective heat flux calculation
    q_conv = Ni * h * Vi  # W/m^2 assuming area of flow is 1 m^2

    return q_conv

def divq_integrate(dataframe, snb_int = False):
    """
    Calculate the total heat flux from the divergence of the Spitzer-Harm fluxes.
    If snb == True outputs integral of divq_snb, otherwise outputs integral of divq_sh.
    """

    x = dataframe['y']
    Te = dataframe['Te']
    div_q_snb = dataframe['Div_Q_SNB']
    div_q_sh = dataframe['Div_Q_SH']
    q_snb = cumtrapz(div_q_snb, x, initial=0)
    q_sh = cumtrapz(div_q_sh, x, initial=0)

    if snb_int == False:
        return q_sh
    else:
        return q_snb




if __name__ == "__main__":
    print('This is a module, import it in your script.')

    ds = pd.read_pickle('/home/userfs/j/jlb647/w2k/lloyd_sim/hermes-3_simulations/analysis/scripts/flux_limiter_detachment/notebooks/2024-04-flux_limiter_analysis/Flux_limiter_detachment_ITER_final.pickle')

    sh = ds[(ds['alpha'] == 'SH') & (ds['neon_frac'] == 0.0)]

    print(sh['kappa_par_e'])

    x = sh['y']

    q_electron = spitzer_q_electron(sh)

    q_electron_simple = spitzer_q_electron_simple(sh)

    # q_ion = spitzer_q_ion_chat(sh)


    print(q_electron_simple)


    



