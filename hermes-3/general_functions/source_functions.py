from boututils.datafile import DataFile
from boutdata.collect import collect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import platform
import traceback
import xarray as xr
import xbout
from pathlib import Path
import xhermes as xh
import matplotlib.pyplot as plt
import os, sys, pathlib
import numpy as np
import xarray as xr
import xhermes as xh

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/sdtools"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions"))

from convergence_functions import *

from matplotlib.ticker import LogFormatter

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/sdtools"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/transients"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions"))


from plotting_functions import *
from convergence_functions import * 

from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
from hermes3.fluxes import *

def replace_guards(var):
    """
    Replace the points in the guard cells with boundary values.
    """

    var = var[1:-1]  # Strip the edge guard cells

    var[0] = 0.5 * (var[0] + var[1])
    var[-1] = 0.5 * (var[-1] + var[-2])
    
    return var

def pi_feedback_source(ds, plot = False, time_slices = 10, save = False):
    t = ds['t'].values*1e3
    y = ds['y'].values

    upstream_source = ds['Sd+_feedback'].values.reshape(len(t), len(y))

    first_index = upstream_source[:, 0]

    steady_state = first_index[-10:].mean()
    print(f'steady state pi source = {steady_state}')

    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, first_index, label = 'First index')
        ax.axhline(steady_state, color = 'red', label = f'Steady state mean (last {time_slices} steps)', linestyle = '--')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Source')
        ax.set_title(f'Feedback source at first index (average of {time_slices} = {steady_state})')
        ax.legend()
        plt.show()
        if save:
            fig.savefig('PI_source_value.png')

import matplotlib.pyplot as plt

def sum_source(ds, sources=["Te"], sinks=[], plot=False, log_scale=False):
    """
    Sum specified sources and sinks across the spatial domain over time to check 
    for particle balance in steady state, and compute the net total source.

    Parameters:
        ds (xarray.Dataset): The dataset containing simulation data.
        sources (list of str): List of parameter names to sum as sources (positive).
        sinks (list of str): List of parameter names to sum as sinks (negative).
        plot (bool): Whether to plot the results. Default is False.
        log_scale (bool): Whether to use a logarithmic y-scale for the plot.
                          Default is False.
    
    Returns:
        dict: Contains the time-summed values for each source, sink, and the total net source.
    """
    
    # Convert time to milliseconds
    t = ds["t"].values * 1e3
    t -= t[0]
    
    # Compute the volume element (assuming dx, dy, dz, and J are consistent across y)
    volume_element = ds["dx"] * ds["dy"] * ds["dz"] * ds["J"]
    
    # Dictionary to store results for each parameter and the total
    results = {}
    total_source = 0  # Initialize total source to accumulate contributions
    
    # Process sources (positive terms)
    for source_param in sources:
        if source_param not in ds:
            raise ValueError(f"Source parameter '{source_param}' not found in the dataset.")
        
        # Squeeze out singleton dimensions and compute volume-weighted sum
        source = np.abs(ds[source_param].squeeze())
        time_sum = (source * volume_element).sum(dim="y")
        
        # Accumulate into total source as positive
        total_source += time_sum
        
        # Store individual source results
        results[source_param] = time_sum
    
    # Process sinks
    for sink_param in sinks:
        if sink_param not in ds:
            raise ValueError(f"Sink parameter '{sink_param}' not found in the dataset.")
        
        # Squeeze out singleton dimensions and compute volume-weighted sum
        sink = ds[sink_param].squeeze()
        time_sum = (sink * volume_element).sum(dim="y")
        
        if log_scale:
            # Take the absolute value for logging but subtract from total
            abs_time_sum = np.abs(time_sum)
            total_source -= abs_time_sum
            results[sink_param] = abs_time_sum  # Store as positive for plotting
        else:
            # Use negative value for sinks and add to total
            negative_time_sum = -np.abs(time_sum)
            total_source += negative_time_sum  # Accumulate as negative
            results[sink_param] = negative_time_sum  # Store as negative for plotting

    # Store the total source/sink in the results
    results['Total Source'] = total_source

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        
        for parameter, time_sum in results.items():
            if parameter in sources:
                plt.plot(t, time_sum.values, label=f'{parameter} (source)')
            elif parameter in sinks:
                plt.plot(t, time_sum.values, label=f'{parameter} (sink)', linestyle='--')
        
        plt.plot(t, total_source.values, label='Total Source', color='black', linestyle=':')
        
        if log_scale:
            plt.yscale('log')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Source/Sink Sum (s^-1)")
        plt.legend()
        plt.show()
    
    return results


def spatial_sources(ds, source_terms, sink_terms, log_scale = False):
    y = ds['y'].values[1:-1]
    total = np.zeros_like(ds['Sd_target_recycle'].isel(t=-1).values.squeeze()[1:-1])
    fig, ax = plt.subplots(figsize=(10, 6), dpi = 500)
    if source_terms != ['']:
        for i in source_terms:
            if i == 'Sd_target_recycle':
                data = np.abs(ds[i].isel(t=-1).values.squeeze())[2:]
            else:    
                data = np.abs(replace_guards(ds[i].isel(t=-1).values.squeeze()))
            ax.plot(y, data, label = f'{i}')
            total += data

    if sink_terms != ['']:
        for i in sink_terms:
            if i == 'Sd_target_recycle':
                data = np.abs(ds[i].isel(t=-1).values.squeeze())[2:]
                print(data)
            else:
                data = np.abs(replace_guards(ds[i].isel(t=-1).values.squeeze())) * -1
            total += data
            if log_scale:
                data = np.abs(data)
            ax.plot(y, data, label = f'{i}', linestyle = '--')
    ax.plot(y, total, label = 'Total Source', linestyle = ':')
    if log_scale:
        ax.set_yscale('log')
        # ax.set_ybound(1e1, 1e28)
    ax.set_xlabel(r'S$_\parallel$')
    ax.set_ylabel('source (m^_3 s^-1)')
    ax.legend()

    return ax


from scipy.integrate import quad

def q_electron(ds):
    y = ds['y'].values
    t = ds['t'].values # Convert time to milliseconds
    kappa_e = ds['kappa_par_e'].values.reshape(len(t), len(y))
    Te = ds['Te'].values.reshape(len(t), len(y))

    # Apply replace_guards to each time slice
    kappa_e = np.apply_along_axis(replace_guards, axis=1, arr=kappa_e)
    Te = np.apply_along_axis(replace_guards, axis=1, arr=Te)

    # Calculate the gradient of Te along the y-axis
    grad_T = np.gradient(Te, axis=1)
    q = -kappa_e * grad_T

    return q

def ELM_power(t, ELM_fluence, tau_rise):
    """
    Calculate the ELM power profile.

    Parameters:
    t (array-like): Time points for the calculation.
    ELM_fluence (float): Total energy fluence of the ELM.
    tau_rise (float): Rise time of the ELM.

    Returns:
    array-like: The ELM power profile at times `t`.
    """
    tau_ELM = 3 * tau_rise  # Total ELM duration

    def ELM_shape_scalar(t_scalar, tau_rise):
        """
        ELM shape function for a scalar time value.
        """
        tau = tau_rise * 0.8
        epsilon_min = 1e-9
        t_current = max(t_scalar, epsilon_min)  # Time since ELM pulse started
        return (1 + (tau / t_current)**2) * (tau / t_current)**2 * np.exp(-(tau / t_current)**2)

    # Integrate the ELM shape function to calculate the normalisation factor
    I, _ = quad(ELM_shape_scalar, 1e-9, tau_ELM, args=(tau_rise,))

    # Calculate the normalisation constant
    q_0 = ELM_fluence / I
    print(f"q_0 = {q_0} W/m^2")

    # Calculate the ELM power profile for the array `t`
    def ELM_shape_array(t_array):
        """
        ELM shape function for an array of time values.
        """
        tau = tau_rise * 0.8
        epsilon_min = 1e-9
        t_current = np.maximum(t_array, epsilon_min)  # Time since ELM pulse started
        return (1 + (tau / t_current)**2) * (tau / t_current)**2 * np.exp(-(tau / t_current)**2)

    q_ELM = q_0 * ELM_shape_array(t)

    return q_ELM




def ELM_prefactor(base_power, t, ELM_fluence, tau_rise):

    q_ELM = ELM_power(t, ELM_fluence, tau_rise)
    prefactor = base_power/base_power * (1 + q_ELM/(2*base_power))
    return prefactor


if __name__ == '__main__':
    print('pumpkin')
    # ds = xh.load_hermes3_data('/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/feedback/feedback_1/feedback_1.nc')
    # pi_feedback_source(ds, plot = True)