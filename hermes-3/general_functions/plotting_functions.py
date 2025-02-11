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

import matplotlib.animation as animation
from matplotlib.ticker import LogFormatter
# from moviepy.editor import VideoFileClip

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

import matplotlib.pyplot as plt
import numpy as np

def plot_profiles(simulation_data, variables=['Te'], data_label=None,
                  guard_replace=True, linestyles=None, log_threshold=1e6, save=False):
    """
    Plots specified variable profiles for the given simulation data.

    Parameters:
    simulation_data (xarray Dataset): Dataset for the simulation.
    variables (list): List of variables to plot (e.g., ['Te', 'Ti']).
    data_label (str, optional): Label for the dataset in the plot legend.
    guard_replace (bool): Whether to replace guard cells.
    linestyles (list, optional): Custom linestyles for each variable plot.
    log_threshold (float): Threshold above which the y-axis will be plotted in log scale.
    save (bool): Whether to save the plot as an image file.
    
    Returns:
    list of Axes: List of matplotlib Axes objects for further manipulation.
    """
    simulation_data = simulation_data.isel(t=-1)  # Select the last time step
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

    for i, var in enumerate(variables):
        ax = axs[i]

        y = simulation_data['y'].values
        var_data = np.ravel(simulation_data[var].values)

        if guard_replace:
            y = y[1:-1]
            var_data = replace_guards(var_data)

        label = f'({var})'
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
        units = simulation_data[var].attrs.get('units', 'Unknown units')

        ax.set_xlabel(r'S$_\parallel$ (m)')
        ax.set_ylabel(f'{var} ({units})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

    # Hide any unused axes (if num_vars is odd)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if save:
        # Save the profiles plot to the current directory
        profiles_filename = "profiles_plot.png"
        plt.savefig(profiles_filename)
        print(f"Profiles plot saved as {profiles_filename}")
        plt.close(fig)  # Close to avoid displaying in interactive environments

    return axs  # Return all axes for further manipulation





def compare_profiles(datasets, variables=['Te'], guard_replace=True, linestyles=None):
    """
    Compares multiple variable profiles across datasets.

    Parameters:
    datasets (dict): Datasets passed as keyword arguments for comparison.
    variables (list): List of variables to plot (e.g., ['Te', 'Ti']).
    guard_replace (bool): Whether to replace guard cells in the data.
    linestyles (list, optional): List of line styles for each dataset. If None, 
                                 a default style is applied.

    Returns:
    axs (list): List of axes with the plotted data.
    """
    num_datasets = len(datasets)
    num_vars = len(variables)

    # Set up plot layout with two columns if multiple variables are passed
    ncols = 2 if num_vars > 1 else 1
    nrows = (num_vars + 1) // 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), dpi=500)
    axs = np.ravel(axs)  # Flatten the axes array for easy indexing

    if linestyles is None:
        linestyles = ['-'] * num_datasets  # Default linestyle if not provided

    for i, var in enumerate(variables):
        ax = axs[i]

        for j, (label, dataset) in enumerate(datasets.items()):
            dataset = dataset.isel(t=-1)  # Select the last time step
            y = dataset['y'].values
            var_data = np.ravel(dataset[var].values)

            # Ensure guard cells are removed consistently
            if guard_replace:
                y = y[1:-1]
                var_data = replace_guards(var_data)

            # Ensure y and var_data have the same shape
            if len(var_data) > len(y):
                var_data = var_data[:len(y)]
            elif len(var_data) < len(y):
                y = y[:len(var_data)]

            ax.plot(y, var_data, label=f'{label} ({var})', linestyle=linestyles[j])

        # Get units from the first dataset, assuming they are the same across datasets
        units = list(datasets.values())[0][var].attrs.get('units', 'Unknown units')

        ax.set_xlabel(r'S$_\parallel$ (m)')
        ax.set_ylabel(f'{var} ({units})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

    # Hide any unused axes (if the number of variables is odd)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return axs


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


def plot_te_ne_nd(ds):
    
    ds = ds.isel(t=-1)
    Te = replace_guards(np.ravel(ds['Te']))
    Ne = replace_guards(np.ravel(ds['Ne']))
    Nd = replace_guards(np.ravel(ds['Nd']))
    y = ds['y'][1:-1]

    fig,ax = plt.subplots(1,1, figsize = (12,6))
    ax.plot(y,Te, label = 'Te')
    ax.set_ylabel('Te (eV)')
    ax.set_xlabel(rf'S$_\parallel$ (m)')
    ax2 = ax.twinx()

    ax2.plot(y,Ne, label = 'Ne', color = 'red')
    ax2.plot(y,Nd, label = 'Nd', color = 'green', linestyle = '--')
    ax2.set_yscale('log')
    ax2.grid(False)
    ax2.set_ylabel(rf'Ne, Nd (m$^-3$)')
    fig.legend()

    return ax



def log_formatter():
    """
    Creates and returns a LogFormatter for use in plotting log-scaled data.
    
    This formatter will format the y-axis ticks to show non-scientific notation
    when the values are powers of 10, and scientific notation for others.

    Returns:
    LogFormatter: The formatter for logarithmic scale.
    """
    return LogFormatter(base=10.0, labelOnlyBase=False)


def plot_profiles_animation(simulation_data, variables=['Te'], data_label=None,
                            guard_replace=True, linestyles=None, log_threshold=1e6, filename='profiles_animation.gif',
                            time_slices=(-40, None), step=1, detachment_front=False):
    """
    Creates an animated GIF of the specified variable profiles for a user-specified range of time slices,
    with time plotted in milliseconds. Optionally, plot the detachment front location.

    Parameters:
    simulation_data (xarray Dataset): Dataset for the simulation.
    variables (list): List of variables to plot (e.g., ['Te', 'Ti']).
    data_label (str, optional): Label for the dataset in the plot legend.
    guard_replace (bool): Whether to replace guard cells.
    linestyles (list, optional): Custom linestyles for each variable plot.
    log_threshold (float): Threshold above which the y-axis will be plotted in log scale.
    filename (str): The filename to save the animation as a GIF.
    time_slices (tuple): A tuple specifying the range of time slices to use for the animation (start_slice, end_slice).
                         Defaults to the last 40 time slices.
    step (int): Step interval for sampling time slices to reduce the number of frames.
    detachment_front (bool): If True, plot the detachment front as a horizontal dashed red line.
    """
    # Unpack the start and end of the time slice range
    start_slice, end_slice = time_slices

    # Select the time slices from the dataset with the specified step interval
    selected_steps = simulation_data.isel(t=slice(start_slice, end_slice, step))
    num_timesteps = selected_steps.sizes['t']  # Calculate the number of timesteps in the sampled range
    num_vars = len(variables)

    # Get time values in seconds and convert to milliseconds
    times = selected_steps['t'].values  # Corresponding time values
    # t_conversion = simulation_data['t'].attrs.get('conversion', 1.0)
    times = times * 1e3  # Convert to milliseconds
    times = times - times[0]  # Normalize time to start from zero

    # Set up plot layout with two columns, adjusting rows based on the number of variables
    ncols = 2 if num_vars > 1 else 1
    nrows = (num_vars + 1) // 2  # Ensure enough rows

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows), dpi=500)
    
    # If we have only one subplot, axs won't be a list, so we ensure it's treated as such
    if num_vars == 1:
        axs = [axs]

    # Flatten axs in case of multiple rows and columns, to handle indexing uniformly
    axs = np.ravel(axs)

    if linestyles is None:
        linestyles = ['-'] * num_vars  # Default linestyle if not provided

    # Precompute y-axis limits
    y_min = {}
    y_max = {}
    for var in variables:
        data_var = selected_steps[var]
        if guard_replace:
            data_var = data_var.isel(y=slice(1, -1))  # Exclude guard cells if needed
        y_min[var] = data_var.min().values
        y_max[var] = data_var.max().values

    # Calculate initial detachment front position for static purple line
    initial_detachment_front = None
    detachment_front_max = None

    if detachment_front:
        # Determine the initial detachment front for the static purple line
        initial_detachment_front = detachment_front_finder(selected_steps.isel(t=0))

        # Calculate the maximum detachment front position over all time steps
        detachment_front_max = max(
            detachment_front_finder(selected_steps.isel(t=t_index)) or 0
            for t_index in range(num_timesteps)
        )

    def update_plot(t_index):
        """Updates the plot for the given time index."""
        current_data = selected_steps.isel(t=t_index)  # Select the time step within the sampled range
        current_time_ms = times[t_index]  # Get the corresponding time in milliseconds

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
            ax.set_title(f'Time: {current_time_ms:.2f} ms')

            ax.set_ylim(y_min[var], y_max[var])  # Set fixed y-axis limits

            # Set xlim based on maximum detachment front position with a 10% buffer
            if detachment_front and detachment_front_max is not None:
                buffer = detachment_front_max * 0.1
                ax.set_xlim(max(y) - detachment_front_max - buffer, max(y))

                # Plot the static initial detachment front as a purple line
                if initial_detachment_front is not None:
                    ax.axvline(initial_detachment_front, color='purple', linestyle='-', label="Initial Detachment Front")

                # Plot dynamic detachment front as a red dashed line for each frame
                front_loc = detachment_front_finder(current_data)
                if front_loc is not None:
                    ax.axvline(front_loc, color='red', linestyle='--', label=f"Detachment Front: {front_loc:.2f} m")

    # Create animation using FuncAnimation
    ani = animation.FuncAnimation(fig, update_plot, frames=num_timesteps, repeat=False)

    # Save the animation as a GIF using PillowWriter
    ani.save(filename, writer='pillow', fps=10)

    print(f"Animation saved as {filename}")
    plt.close()



def convert_gif_to_mp4(gif_path, mp4_path):
    """
    Converts a GIF file to an MP4 file.

    Parameters:
    gif_path (str): Path to the input GIF file.
    mp4_path (str): Path to save the output MP4 file.
    """
    # Load the GIF
    clip = VideoFileClip(gif_path)
    
    # Write to MP4 format
    clip.write_videofile(mp4_path, codec="libx264")


def particle_sink_source(ds, sources, sinks, variable=None, log_scale=False):
    """
    Plots the individual particle sources and sinks over time and calculates
    the net balance as total sources minus total sinks, or source - sink
    if log_scale is True. Sources are plotted as solid lines, and sinks as
    dashed lines (made positive if log_scale is True).

    Parameters:
    ds (xarray.Dataset): The dataset containing particle source and sink variables.
    sources (list): List of variable names for sources.
    sinks (list): List of variable names for sinks.
    variable (str, optional): An additional variable to plot.
    log_scale (bool): If True, plots the y-axis on a logarithmic scale and
                      source/sink terms are made positive, with total defined
                      as source - sink.
    """
    # Dictionary to store results for each variable
    data_dict = {}

    # Calculate the y intervals for non-uniform spacing
    y = ds['y'].values
    # Extend dy to match the length of y by adding the last interval
    dy = np.concatenate([np.diff(y), [y[-1] - y[-2]]])

    # Calculate the total particles for sources by summing density * interval width
    for var in sources:
        if 't' in ds[var].dims:
            # Multiply density by interval width (dy) across the full range
            data_dict[var] = (abs(ds[var] * dy).sum(dim=[dim for dim in ds[var].dims if dim != 't']).values)
            print(f'{var} : {data_dict[var]}')
        else:
            data_dict[var] = (abs(ds[var] * dy).values.sum())

    # Calculate the total particles for sinks by summing density * interval width
    # Make sinks positive if log_scale is True
    for var in sinks:
        if 't' in ds[var].dims:
            value = (abs(ds[var] * dy).sum(dim=[dim for dim in ds[var].dims if dim != 't']).values)
            data_dict[var] = value if log_scale else -value
            print(f'{var} : {data_dict[var]}')
        else:
            value = (abs(ds[var] * dy).values.sum())
            data_dict[var] = value if log_scale else -value

    # Add an optional variable if provided
    if variable is not None:
        units = ds[variable].attrs.get('units', 'Unknown units')
        data_dict[variable] = (ds[variable]).sum(dim=[dim for dim in ds[variable].dims if dim != 't']).values

    # Convert dictionary to DataFrame, using the time values as the index
    df_d = pd.DataFrame(data_dict, index=(ds['t'].values * 1e3))  # Convert time to ms

    # Calculate the total source as source - sink if log_scale, otherwise source + sink
    total_sources = df_d[sources].sum(axis=1)
    total_sinks = df_d[sinks].sum(axis=1)
    if log_scale:
        df_d["Total Source - Sinks"] = total_sources - total_sinks
        print(f'Total Source - Sinks : {df_d["Total Source - Sinks"]}')
    else:
        df_d["Total Source - Sinks"] = total_sources + total_sinks
        print(f'Total Source + Sinks : {df_d["Total Source - Sinks"]}')

    # Plot each source and sink with specific line styles
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=500)
    for column in df_d.columns:
        if column in sources:
            ax.plot(df_d.index, df_d[column], linestyle='-', label=column)  # Solid line for sources
        elif column in sinks:
            ax.plot(df_d.index, df_d[column], linestyle='--', label=column)  # Dashed line for sinks

    # Plot the net balance
    ax.plot(df_d.index, df_d["Total Source - Sinks"], linestyle='-', color='black', label='Total Source - Sinks')

    # Plot an additional variable if provided
    if variable is not None:
        ax_twin = ax.twinx()
        ax_twin.plot(df_d.index, df_d[variable], linestyle=':', color='black', label=f'{variable}')
        ax_twin.set_ylabel(f'Spatial Sum of {variable} ({units})')
        ax_twin.set_yscale('log' if log_scale else 'linear')

    # Set y-axis scale for the main plot
    if log_scale:
        ax.set_yscale("log")

    # Labels and legend
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Particle Source/Sink (s^-1)")
    ax.grid(True)
    ax.legend(loc='best', fontsize=8)
    fig.show()
