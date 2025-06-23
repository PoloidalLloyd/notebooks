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
import imageio.v2 as imageio  # Use this to avoid deprecation warnings
import tempfile
from matplotlib.ticker import ScalarFormatter
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


def detachment_front_finder(ds, last_time_slice=True, use_temperature=True):
    """
    Finds the location of the detachment front. Optionally, it can be determined
    by the first cell where Te <= 5.

    Parameters:
    ds (xarray Dataset): The dataset for a single time slice.
    use_temperature (bool): If True, the front is determined by the first cell where Te <= 5.
                            If False, the front is determined where Nd > Ne.

    Returns:
    float: The y-coordinate of the detachment front, 
           or zero if the front position is undefined or non-positive.
    """
    Nd = replace_guards(np.ravel(ds['Nd']))
    Ne = replace_guards(np.ravel(ds['Ne']))
    Te = replace_guards(np.ravel(ds['Te']))  # Adding temperature field
    y = ds['y'][1:-1]  # Exclude guards from y coordinate

    if use_temperature:
        # Find the first index where Te <= 5
        detachment_indices = np.where(Te <= 5)[0]
    else:
        # Find the first index where Nd > Ne
        detachment_indices = np.where(Nd > Ne)[0]

    if len(detachment_indices) > 0:
        front_loc = y[detachment_indices[0]]
        front_position = y[-1].values - front_loc.values  # Relative to y-max
        return max(front_position, 0)  # Ensure non-negative output
    else:
        return 0  # Set to zero if detachment front not found or undefined


def plot_time_history(dataset, variables=['Te'], upstream_index=2, target_index=-2,
                      track_detachment_front=False, time_slices=800,
                      log_threshold=1e6, base_figsize=(6, 4), save=False,
                      det_specification='Te', source_totals=False, source_variables=None):
    if source_variables is None:
        source_variables = []

    num_time_slices = min(time_slices, dataset.sizes['t'])
    selected_steps = dataset.isel(t=slice(-num_time_slices, None))
    times = selected_steps['t'].values * 1e3  # ms
    last_time_step = times[-1]

    n_cols = max(len(variables), len(source_variables), 1)
    n_rows = 2 + int(source_totals) + int(track_detachment_front)

    figsize = (base_figsize[0] * n_cols, base_figsize[1] * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=200)

    # Always return 2D array
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)
    axs = np.array(axs)

    # Compute detachment front if requested
    detachment_front_positions = None
    if track_detachment_front:
        front_positions = []
        for t_step in range(num_time_slices):
            ds_at_t = selected_steps.isel(t=t_step)
            front_loc = detachment_front_finder(ds_at_t, use_temperature=(det_specification == 'Te'))
            front_positions.append(front_loc)
        detachment_front_positions = np.array(front_positions)

    # Upstream + target plots (rows 0 and 1)
    for i, var in enumerate(variables):
        try:
            upstream_data = np.squeeze(selected_steps[var].isel(y=upstream_index).values)
            target_data = np.squeeze(selected_steps[var].isel(y=target_index).values)
        except:
            upstream_data = np.squeeze(selected_steps[var].isel(pos=upstream_index).values)
            target_data = np.squeeze(selected_steps[var].isel(pos=target_index).values)

        scale = "log" if np.max(np.abs([upstream_data, target_data])) > log_threshold else "linear"

        axs[0, i].plot(times, upstream_data, label=f'Upstream {var}', marker='o', linestyle='-')
        axs[0, i].set_title(f'Upstream {var}')
        axs[0, i].set_yscale(scale)
        axs[0, i].set_xlabel('Time (ms)')
        axs[0, i].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[0, i].grid(True)

        axs[1, i].plot(times, target_data, label=f'Target {var}', marker='x', linestyle='--')
        axs[1, i].set_title(f'Target {var}')
        axs[1, i].set_yscale(scale)
        axs[1, i].set_xlabel('Time (ms)')
        axs[1, i].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
        axs[1, i].grid(True)

    # Spatial totals (row 2 if enabled)
    if source_totals:
        for i, var in enumerate(source_variables):
            totals = [np.sum(np.squeeze(selected_steps[var].isel(t=t_step).values)) for t_step in range(num_time_slices)]
            axs[2, i].plot(times, totals, label=f'Total {var}', marker='s', linestyle='-')
            axs[2, i].set_title(f'Spatial Total {var}')
            axs[2, i].set_xlabel('Time (ms)')
            axs[2, i].set_ylabel(f'{var} ({dataset[var].attrs.get("units", "Unknown units")})')
            axs[2, i].grid(True)

    # Detachment front (last row, last column only)
    if track_detachment_front:
        row_idx = n_rows - 1
        col_idx = n_cols - 1
        label = 'Te <= 5 Front' if det_specification == 'Te' else 'Nd > Ne Front'

        axs[row_idx, col_idx].plot(times, detachment_front_positions, color='red', marker='s', linestyle='-')
        axs[row_idx, col_idx].set_title(f'Detachment Front Position ({label})')
        axs[row_idx, col_idx].set_xlabel('Time (ms)')
        axs[row_idx, col_idx].set_ylabel('Position (m)')
        axs[row_idx, col_idx].grid(True)

    # Hide unused axes
    for r in range(n_rows):
        for c in range(n_cols):
            if (r == 0 and c < len(variables)) or \
               (r == 1 and c < len(variables)) or \
               (r == 2 and source_totals and c < len(source_variables)) or \
               (r == n_rows - 1 and track_detachment_front and c == n_cols - 1):
                continue  # in use
            axs[r, c].axis('off')

    plt.suptitle(f"Time History of Variables (Last time step: {last_time_step:.8f} ms / {last_time_step*1e-3:.8f} s)", fontsize=16)
    plt.tight_layout()

    if save:
        filename = "time_history_plot.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_profiles_animation(simulation_data, variables=['Te'], data_label=None,
                            guard_replace=True, linestyles=None, log_threshold=1e3,
                            filename='profiles_animation.mp4', max_frames=40, fps=3):
    """
    Creates an animated video of the specified variable profiles for up to `max_frames` time steps.

    Parameters:
    simulation_data (xarray Dataset): Dataset for the simulation.
    variables (list): List of variables to plot (e.g., ['Te', 'Ti']).
    data_label (str, optional): Label for the dataset in the plot legend.
    guard_replace (bool): Whether to replace guard cells.
    linestyles (list, optional): Custom linestyles for each variable plot.
    log_threshold (float): Threshold above which the y-axis will be plotted in log scale.
    filename (str): The filename to save the animation as a video (e.g., `.mp4` or `.gif`).
    max_frames (int): Maximum number of frames to use in the animation.
    fps (int): Frames per second for the output video.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import imageio
    from PIL import Image
    import math

    linestyles = linestyles or ['-'] * len(variables)
    num_available = simulation_data.dims['t']
    num_frames = min(max_frames, num_available)

    time_indices = np.linspace(-num_frames, -1, num_frames, dtype=int)

    output_dir = "./frames"
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []

    # Determine subplot grid layout
    n_vars = len(variables)
    n_cols = math.ceil(math.sqrt(n_vars))
    n_rows = math.ceil(n_vars / n_cols)

    for frame_idx, t_idx in enumerate(time_indices):
        print(f"Generating frame {frame_idx + 1}/{num_frames} (t index = {t_idx})")
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=100)
        axs = np.array(axs).reshape(-1)  # Flatten in case of single row/column

        current_data = simulation_data.isel(t=t_idx)

        for i, var in enumerate(variables):
            ax = axs[i]
            y = current_data['y'].values
            var_data = np.ravel(current_data[var].values)

            if guard_replace:
                y = y[1:-1]
                var_data = replace_guards(var_data)

            label = f'{data_label or ""} ({var})'
            ax.plot(y[::-1], var_data, label=label, linestyle=linestyles[i])

            if np.max(np.abs(var_data)) > log_threshold:
                    if np.any(var_data < 0):
                        ax.set_yscale('symlog', linthresh=1e-3)
                        ax.yaxis.set_major_formatter(ScalarFormatter())
                    else:
                        ax.set_yscale('log')
                        ax.yaxis.set_major_formatter(ScalarFormatter())
            else:
                ax.set_yscale('linear')

            ax.set_xscale('log')
            units = current_data[var].attrs.get('units', 'Unknown units')
            ax.set_xlabel('S$_\\parallel$ (m)')
            ax.set_ylabel(f'{var} ({units})')
            ax.set_title(f'Time step {frame_idx + 1}/{num_frames} \n time = {simulation_data["t"].values[t_idx]*1e3:.2f} (ms)')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True)

        # Turn off any unused subplots
        for j in range(len(variables), len(axs)):
            axs[j].axis('off')

        fig.tight_layout()
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
        fig.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)

    # Ensure consistent frame size
    first_frame = imageio.imread(frame_paths[0])
    target_size = (first_frame.shape[1], first_frame.shape[0])

    if filename.endswith('.mp4'):
        writer = imageio.get_writer(filename, fps=fps, codec='libx264', format='ffmpeg')
    else:
        writer = imageio.get_writer(filename, fps=fps)

    for path in frame_paths:
        frame = imageio.imread(path)
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = np.array(Image.fromarray(frame).resize(target_size, Image.BICUBIC))
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        writer.append_data(frame)

    writer.close()
    print(f"Animation saved to {filename}")



if __name__ == '__main__':
    # do something?
    print("Hello world!")