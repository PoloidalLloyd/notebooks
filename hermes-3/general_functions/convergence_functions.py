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
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions/plotting"))

from plotting_functions import *

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


def find_first_below_threshold(temp_profile, y_values, threshold=5.0):
    """
    Finds the first location where the temperature drops below a given threshold.

    Parameters:
    temp_profile (np.array): The temperature profile along the spatial axis.
    y_values (np.array): The spatial locations corresponding to the temperature values.
    threshold (float): The temperature threshold to check against (default is 5 eV).

    Returns:
    float: The y-position where the temperature first drops below the threshold, or None if not found.
    """
    below_threshold = np.where(temp_profile < threshold)[0]
    
    if len(below_threshold) > 0:
        return y_values[below_threshold[0]]  # Return the first position
    else:
        return None  # Return None if no location is found below the threshold


def detachment_front_finder(ds, last_time_slice = True):
    """
    Finds the location where Nd becomes greater than Ne in the dataset.
    
    Parameters:
    ds (xarray Dataset): The dataset for a single time slice.
    
    Returns:
    float: The y-coordinate of the detachment front (where Nd > Ne).
    """
    Nd = replace_guards(np.ravel(ds['Nd']))
    Ne = replace_guards(np.ravel(ds['Ne']))
    y = ds['y'][1:-1]  # Exclude guards from y coordinate as well
    
    front_loc = y[np.where(Nd > Ne)[0][0]]
    return y[-1].values - front_loc.values

def plot_time_history(dataset, variables=['Te'], upstream_index=2, target_index=-2, 
                      track_detachment_front=False, time_slices=200):
    """
    Plots the time history of user-specified variables at upstream and target positions
    on separate plots, using the last 200 time slices or the maximum available.
    
    Optionally tracks the detachment front, where Nd > Ne, and adds it as a separate subplot.

    Parameters:
    dataset (xarray Dataset): Hermes-3 dataset.
    variables (list): List of variables to plot (e.g., ['Te', 'Td+', 'Ne']).
    upstream_index (int): Index for the upstream data.
    target_index (int): Index for the target data.
    track_detachment_front (bool): If True, track the location where Nd > Ne
                                   and show it as a separate subplot.
    """
    # Determine how many time steps to plot (maximum 200 or the total available)
    num_time_slices = min(time_slices, dataset.sizes['t'])
    
    # Select the last `num_time_slices` time steps
    selected_steps = dataset.isel(t=slice(-num_time_slices, None))
    times = selected_steps['t'].values  # Corresponding time values
    t_conversion = dataset['t'].attrs['conversion']
    times = times * t_conversion  # Convert to seconds


    # Adjust the number of subplots based on whether we're tracking detachment front
    total_vars = len(variables) + (1 if track_detachment_front else 0)

    # Create figure with subplots, one set for upstream and one for target, plus detachment front if enabled
    fig, axs = plt.subplots(2, total_vars, figsize=(12, 6), dpi=500)

    # Ensure axs is treated as a list if there's only one plot
    if total_vars == 1:
        axs = [axs[0], axs[1]]

    axs = np.ravel(axs)  # Flatten the axes for easy handling

    # Variable to store positions where Nd > Ne (detachment front)
    detachment_front_positions = [] if track_detachment_front else None

    # If tracking the detachment front, calculate it for each time slice
    if track_detachment_front:
        front_positions = []
        
        for t_step in range(num_time_slices):
            ds_at_t = selected_steps.isel(t=t_step)
            front_loc = detachment_front_finder(ds_at_t, last_time_slice=False)
            front_positions.append(front_loc)

        detachment_front_positions = np.array(front_positions)

    # Iterate over each variable to plot upstream and target values
    for i, var in enumerate(variables):
        y_values = selected_steps['y'].values

        # Extract upstream and target data for each variable
        upstream_data = np.squeeze(selected_steps[var].isel(y=upstream_index).values)
        target_data = np.squeeze(selected_steps[var].isel(y=target_index).values)

        # Plot upstream data on the top row
        axs[i].plot(times, upstream_data, label=f'Upstream {var}', marker='o', linestyle='-')
        axs[i].set_title(f'Upstream {var}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[i].grid(True)

        # Plot target data on the bottom row
        axs[i + total_vars].plot(times, target_data, label=f'Target {var}', marker='x', linestyle='--')
        axs[i + total_vars].set_title(f'Target {var}')
        axs[i + total_vars].set_xlabel('Time (s)')
        axs[i + total_vars].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[i + total_vars].grid(True)

    # Add a separate subplot for the detachment front position if requested
    if track_detachment_front:

        # Plot the tracked detachment front position over time in the last subplot
        detachment_front_index = len(variables)  # The next index after all variables
        axs[detachment_front_index].plot(times, detachment_front_positions, marker='s', linestyle='-', color='red', 
                                         label='Nd > Ne Front')
        axs[detachment_front_index].set_title('Detachment Front Position (Nd > Ne)')
        axs[detachment_front_index].set_xlabel('Time (s)')
        axs[detachment_front_index].set_ylabel('Position (m)')
        axs[detachment_front_index].grid(True)

        # Plot the same on the bottom row
        axs[detachment_front_index + total_vars].plot(times, detachment_front_positions, marker='s', linestyle='-', color='red', 
                                                      label='Nd > Ne Front')
        axs[detachment_front_index + total_vars].set_title('Detachment Front Position (Nd > Ne)')
        axs[detachment_front_index + total_vars].set_xlabel('Time (s)')
        axs[detachment_front_index + total_vars].set_ylabel('Position (m)')
        axs[detachment_front_index + total_vars].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # do something?
    print("Hello world!")