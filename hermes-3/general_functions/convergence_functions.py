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
import matplotlib.animation as animation
from matplotlib.ticker import LogFormatter
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


def log_formatter():
    """
    Creates and returns a LogFormatter for use in plotting log-scaled data.
    
    This formatter will format the y-axis ticks to show non-scientific notation
    when the values are powers of 10, and scientific notation for others.

    Returns:
    LogFormatter: The formatter for logarithmic scale.
    """
    return LogFormatter(base=10.0, labelOnlyBase=False)


def replace_guards(var):
    """
    This in-place replaces the points in the guard cells with the points 
    on the boundary.
    """
    var = var[1:-1]  # Strip the edge guard cells

    var[0] = 0.5 * (var[0] + var[1])
    var[-1] = 0.5 * (var[-1] + var[-2])
    
    return var

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


def detachment_front_finder(ds, last_time_slice=True):
    """
    Finds the location where Nd becomes greater than Ne in the dataset.

    Parameters:
    ds (xarray Dataset): The dataset for a single time slice.

    Returns:
    float: The y-coordinate of the detachment front (where Nd > Ne), 
           or zero if the front position is undefined or non-positive.
    """
    Nd = replace_guards(np.ravel(ds['Nd']))
    Ne = replace_guards(np.ravel(ds['Ne']))
    y = ds['y'][1:-1]  # Exclude guards from y coordinate as well

    detachment_indices = np.where(Nd > Ne)[0]

    if len(detachment_indices) > 0:
        front_loc = y[detachment_indices[0]]
        front_position = y[-1].values - front_loc.values  # Relative to y-max
        return max(front_position, 0)  # Ensure non-negative output
    else:
        return 0  # Set to zero if detachment front not found or undefined

       
def plot_time_history(dataset, variables=['Te'], upstream_index=2, target_index=-2,
                      track_detachment_front=False, time_slices=800,
                      log_threshold=1e6, base_figsize=(6, 4), save=False):
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
    log_threshold (float): Threshold above which the y-axis will be plotted in log scale.
    base_figsize (tuple): Base figure size for a single plot (width, height).
    """
    # Determine how many time steps to plot (maximum 200 or the total available)
    num_time_slices = min(time_slices, dataset.sizes['t'])

    # Select the last `num_time_slices` time steps
    selected_steps = dataset.isel(t=slice(-num_time_slices, None))
    times = selected_steps['t'].values 
    t_conversion = dataset['t'].attrs.get('conversion', 1.0)
    # times = times * t_conversion  
    times = times * 1e3  # Convert to milliseconds for plotting

    # Find the last time step in milliseconds
    last_time_step = times[-1]

    # Adjust the number of subplots based on whether we're tracking detachment front
    total_vars = len(variables) + (1 if track_detachment_front else 0)

    # Dynamically scale figure size based on number of variables
    figsize = (base_figsize[0] * total_vars, base_figsize[1] * 2)

    # Create figure with subplots, one set for upstream and one for target, plus detachment front if enabled
    fig, axs = plt.subplots(2, total_vars, figsize=figsize, dpi=200)

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
            front_loc = detachment_front_finder(ds_at_t)
            front_positions.append(front_loc)
        detachment_front_positions = np.array(front_positions)

    # Iterate over each variable to plot upstream and target values
    for i, var in enumerate(variables):
        # Extract upstream and target data for each variable
        upstream_data = np.squeeze(selected_steps[var].isel(y=upstream_index).values)
        target_data = np.squeeze(selected_steps[var].isel(y=target_index).values)

        # Check if data exceeds the threshold, and use log scale if so
        if np.max(np.abs(upstream_data)) > log_threshold or np.max(np.abs(target_data)) > log_threshold:
            scale = "log"
        else:
            scale = "linear"

        # Plot upstream data on the top row
        axs[i].plot(times, upstream_data, label=f'Upstream {var}', marker='o', linestyle='-')
        axs[i].set_title(f'Upstream {var}')
        axs[i].set_xlabel('Time (ms)')
        axs[i].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[i].grid(True)
        axs[i].set_yscale(scale)

        # Apply custom log formatter for log scale
        if scale == "log":
            axs[i].yaxis.set_major_formatter(log_formatter())

        # Plot target data on the bottom row
        axs[i + total_vars].plot(times, target_data, label=f'Target {var}', marker='x', linestyle='--')
        axs[i + total_vars].set_title(f'Target {var}')
        axs[i + total_vars].set_xlabel('Time (ms)')
        axs[i + total_vars].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[i + total_vars].grid(True)
        axs[i + total_vars].set_yscale(scale)

        # Apply custom log formatter for target plot
        if scale == "log":
            axs[i + total_vars].yaxis.set_major_formatter(log_formatter())

    # Add a separate subplot for the detachment front position if requested
    if track_detachment_front:
        detachment_front_index = len(variables)  # The next index after all variables
        axs[detachment_front_index].plot(times, detachment_front_positions, marker='s', linestyle='-', color='red',
                                         label='Nd > Ne Front')
        axs[detachment_front_index].set_title('Detachment Front Position (Nd > Ne)')
        axs[detachment_front_index].set_xlabel('Time (ms)')
        axs[detachment_front_index].set_ylabel('Position (m)')
        axs[detachment_front_index].grid(True)

        # Plot the same on the bottom row
        axs[detachment_front_index + total_vars].plot(times, detachment_front_positions, marker='s', linestyle='-', color='red',
                                                      label='Nd > Ne Front')
        axs[detachment_front_index + total_vars].set_title('Detachment Front Position (Nd > Ne)')
        axs[detachment_front_index + total_vars].set_xlabel('Time (ms)')
        axs[detachment_front_index + total_vars].set_ylabel('Position (m)')
        axs[detachment_front_index + total_vars].grid(True)

    # Set the figure's overall title with the time of the last time step
    plt.suptitle(f"Time History of Variables (Last time step: {last_time_step:.8f} (ms) / {last_time_step*1e-3:.8f}(s)", fontsize=16)

    plt.tight_layout()
    print(f"final time step: {last_time_step} (ms) / {last_time_step*1e-3}(s)")

    if save:
        time_history_filename = "time_history_plot.png"
        print(f'Time history plot saved as {time_history_filename}')
        plt.savefig(time_history_filename)
        plt.close()


def plot_profiles_animation(simulation_data, variables=['Te'], data_label=None,
                            guard_replace=True, linestyles=None, log_threshold=1e6, filename='profiles_animation.gif'):
    """
    Creates an animated GIF of the specified variable profiles for the last 20 time steps (or fewer).

    Parameters:
    simulation_data (xarray Dataset): Dataset for the simulation.
    variables (list): List of variables to plot (e.g., ['Te', 'Ti']).
    data_label (str, optional): Label for the dataset in the plot legend.
    guard_replace (bool): Whether to replace guard cells.
    linestyles (list, optional): Custom linestyles for each variable plot.
    log_threshold (float): Threshold above which the y-axis will be plotted in log scale.
    filename (str): The filename to save the animation as a GIF.
    """
    num_timesteps = min(100, simulation_data.dims['t'])  # Use last 20 timesteps or fewer
    num_vars = len(variables)

    # Set up plot layout with two columns, adjusting rows based on the number of variables
    ncols = 2 if num_vars > 1 else 1
    nrows = (num_vars + 1) // 2  # Ensure enough rows

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), dpi=500)
    
    # If we have only one subplot, axs won't be a list, so we ensure it's treated as such
    if num_vars == 1:
        axs = [axs]

    # Flatten axs in case of multiple rows and columns, to handle indexing uniformly
    axs = np.ravel(axs)

    if linestyles is None:
        linestyles = ['-'] * num_vars  # Default linestyle if not provided

    def update_plot(t_index):
        """Updates the plot for the given time index."""
        current_data = simulation_data.isel(t=-num_timesteps + t_index)  # Select the time step

        for i, var in enumerate(variables):
            ax = axs[i]
            ax.clear()  # Clear the previous frame

            y = current_data['y'].values
            var_data = np.ravel(current_data[var].values)

            if guard_replace:
                y = y[1:-1]
                var_data = replace_guards(var_data)

            label = f'{data_label} ({var})'
            ax.plot(y, var_data, label=label, linestyle=linestyles[i])

            # Determine if log scale is needed based on threshold
            if np.max(np.abs(var_data)) > log_threshold:
                scale = "log"
            else:
                scale = "linear"

            # Set the appropriate scale
            ax.set_yscale(scale)
            if scale == "log":
                ax.yaxis.set_major_formatter(log_formatter())  # Apply log formatting

            # Get units
            units = current_data[var].attrs.get('units', 'Unknown units')

            ax.set_xlabel(r'S$_\parallel$ (m)')
            ax.set_ylabel(f'{var} ({units})')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True)
            ax.set_title(f'Time step {t_index + 1}/{num_timesteps}')

    # Create animation using FuncAnimation
    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps, repeat=False)

    # Save the animation as a GIF using PillowWriter
    ani.save(filename, writer='pillow', fps=2)

    print(f"Animation saved as {filename}")
    plt.close()

if __name__ == '__main__':
    # do something?
    print("Hello world!")