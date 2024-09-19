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

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/sdtools"))

from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
from hermes3.fluxes import *


# plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
linewidth = 3
markersize = 15



# plt.style.use('ggplot')
plt.style.use('default')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 16})


def replace_guards(var):
    """
    This in-place replaces the points in the guard cells with the points 
    on the boundary.
    """
    var = var[1:-1]  # Strip the edge guard cells

    var[0] = 0.5 * (var[0] + var[1])
    var[-1] = 0.5 * (var[-1] + var[-2])
    
    return var


def load_simulation_data(base_dir, alpha_values, neon_values, replace_guards = True):
    data = {}
    for alpha in alpha_values:
        for neon in neon_values:
            # Construct the path to the specific simulation
            sim_path = os.path.join(base_dir, f'alpha_{alpha}', f'neon_{neon}')
            try:
                print('loading data for alpha={}, neon={}'.format(alpha, neon))
                # Load the latest timestep (t=-1) using BOUT's load.case_1D method
                ds = Load.case_1D(sim_path).ds.isel(t=-1)
                data[(alpha, neon)] = ds
            except Exception as e:
                print(f"Failed to load data for alpha={alpha}, neon={neon}: {e}")
    return data


def plot_temperature_profiles(simulation_data, guard_replace=True, inlcude_axis=True):
    if inlcude_axis:
        fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=500)
    for (alpha, neon), ds in simulation_data.items():
        try:

            
            y = ds['pos'].values
            Te = ds['Te'].values
            print(y)
            if guard_replace:
                y = y[1:-1]
                Te = replace_guards(Te)


            ax.plot(y, Te, label=f'alpha={alpha}, neon={neon}')
        
        except KeyError:
            print(f"Te or y not found in dataset for alpha={alpha}, neon={neon}")
            continue

    ax.set_xlabel('y (Position)')
    ax.set_ylabel('Te (Temperature)')
    ax.set_title('Temperature Profiles for All Simulations')
    ax.legend(loc='best')
    ax.set_xbound(65,72)
    ax.grid(True)


import matplotlib.pyplot as plt


def plot_target_temperatures(simulation_data, data_label=None, 
                             guard_replace=True, ax=None, alpha_values=None):
    """
    Plots the final (target) temperature against the neon fraction for the 
    given simulation data.

    Parameters:
    simulation_data (dict): Dictionary containing datasets to plot.
    data_label (str, optional): Name of the simulation data to be used in the 
                                label. Defaults to None.
    guard_replace (bool): Whether to replace guard cells in the data.
    ax (matplotlib.axes.Axes): Axis to plot on. If None, a new one is created.
    alpha_values (list, optional): List of alpha values to plot. Defaults to None.

    Returns:
    ax (matplotlib.axes.Axes): The axis with the plotted data.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=500)

    neon_fractions = []
    final_temperatures = []

    for (alpha, neon), ds in simulation_data.items():
        if alpha_values is not None and alpha not in alpha_values:
            continue  # Skip this dataset if it's not in the specified values
           
        try:
            Te = ds['Te'].values
            if guard_replace:
                Te = replace_guards(Te)

            # Get the final temperature (last value of Te)
            final_temperature = Te[-1]

            # Store the neon fraction and final temperature
            neon_fractions.append(neon)
            final_temperatures.append(final_temperature)
        
        except KeyError:
            print(f"Te not found in dataset for alpha={alpha}, neon={neon}")
            continue



    # Plot the final temperature against neon fractions
    label = f'{data_label}: alpha={alpha}' if data_label else f'alpha={alpha}'

    ax.plot(neon_fractions, final_temperatures, 'o-', linestyle='-',
            label=label) # if data_label else "Target Temperatures")

    ax.set_xlabel('Neon Fraction')
    ax.set_ylabel('Final Temperature (Te) (eV)')
    ax.grid(True)
    ax.legend(loc='best', fontsize=8)

    return ax



if __name__=="__main__":
    print("Hello, world!")