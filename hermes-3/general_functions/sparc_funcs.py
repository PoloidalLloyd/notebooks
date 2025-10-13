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

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/sdtool_load_test/sdtools"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/transients"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions"))


from plotting_functions import *
from convergence_functions import * 

from hermes3.case_db import *
from hermes3.casedeck import*
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
from hermes3.fluxes import *
from hermes3.selectors import *
from hermes3.front_tracking import *
# from hermes3.balance1d import *



# Funcs


import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from PIL import Image

def single_profile_plot(ds_in, time, x_log_scale=True, title='None', save=False, save_path=None, focus_target=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    linewidth = 2
    ds = ds_in.ds.copy().isel(t=time)

    ax2 = ax.twinx()
    ax2.plot(ds['y'][::-1][2:-2], ds['Te'][2:-2], color='blue', label='Te', linewidth=linewidth)
    ax2.plot(ds['y'][::-1][2:-2], ds['Td+'][2:-2], color='red', label='Td+', linewidth=linewidth)

    ax.plot(ds['y'][::-1][2:-2], ds['Ne'][2:-2], color='blue', label='Ne', linewidth=linewidth, linestyle='--')
    ax.plot(ds['y'][::-1][2:-2], ds['Nd'][2:-2], color='black', label='Nd', linewidth=linewidth, linestyle='--')

    if x_log_scale:
        ax.set_xscale('log')
    else: 
        ax.set_xscale('linear')


    if focus_target:
        ax.set_xlim(0, 5)
        ax.set_ylim(1e18, 1e25)
        ax2.set_ylim(0, 100)

    ax.set_yscale('log')

    ax.set_xlabel('Distance to target (m)', fontsize=20)
    ax.set_ylabel(r'Ne,Nd (m$^{-3}$)', fontsize=20)
    ax2.set_ylabel('T (eV)', fontsize=20)
    ax2.set_ylim(0, 8000)
    ax.set_ylim(1e18, 1e25)
    ax2.grid(False)
    fig.legend(ncol=4, loc='upper right')
    fig.suptitle(f"{(ds_in.ds['t'].values[time] - ds_in.ds['t'].values[0]) * 1e3 : 6f} ms",
                 fontsize=20, x=0.0, horizontalalignment='left')

    if save:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def make_profile_evolution_video(ds_in, start_time, end_time, fps=5, mp4_filename="profile_evolution.mp4", x_log_scale = True, focus_target=False):
    output_dir = "./frames"
    os.makedirs(output_dir, exist_ok=True)

    # Generate frame plots
    frame_paths = []
    for t in range(start_time, end_time):
        frame_path = os.path.join(output_dir, f"frame_{t:03d}.png")
        single_profile_plot(ds_in, time=t, save=True, save_path=frame_path, x_log_scale=x_log_scale)
        frame_paths.append(frame_path)

    # Get target frame size
    first_frame = imageio.imread(frame_paths[0])
    target_size = (first_frame.shape[1], first_frame.shape[0])  # width, height

    writer = imageio.get_writer(mp4_filename, fps=fps, codec='libx264', format='ffmpeg')

    for path in frame_paths:
        frame = imageio.imread(path)

        # Resize if necessary
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = np.array(Image.fromarray(frame).resize(target_size, Image.BICUBIC))

        # Drop alpha if present
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        writer.append_data(frame)

    writer.close()


import numpy as np
import matplotlib.pyplot as plt


def spatial_sum(ds, var, mult_dv=True, spatial_dim='pos', spatial_slice=slice(2, -2)):
    """
    Function to sum over the spatial dimension of a variable.

    Parameters:
    - ds: xarray.Dataset
    - var: variable name as a string
    - mult_dv: whether to multiply by dv
    - spatial_dim: name of spatial dimension
    - spatial_slice: slice of the spatial dimension to include

    Returns:
    - xarray.DataArray of summed values over time
    """
    if mult_dv:
        tot = ds[var].isel({spatial_dim: spatial_slice}) * ds['dv'].isel({spatial_dim: spatial_slice})
    else:
        tot = ds[var].isel({spatial_dim: spatial_slice})

    return tot.sum(dim=spatial_dim)



def plot_spatial_sum_time_series(cs, vars = [''], mult_dv=True, time_range=None,
                                  spatial_dim='pos', y_scale=None, abs = False,plot_src = False, title=None):
    """
    Function to plot the spatial sum of one or more variables over time.

    Parameters:
    - cs: container with .ds (xarray.Dataset)
    - vars: list of variable names (strings)
    - mult_dv: whether to multiply by dv before summing
    - time_range: tuple (start, end) for time slicing
    - spatial_dim: spatial dimension to sum over
    """
    import matplotlib.pyplot as plt
    linewidth = 2
    markersize = 5
    ds = cs.ds
    if time_range is not None:
        ds = ds.isel(t=slice(*time_range))
    else:
        ds = ds.isel(t=slice(0, -1))

    time = (ds['t'].values - ds['t'].values[0]) *1e3  # normalize time to start at 0

    fig, ax = plt.subplots(figsize=(12, 8))
    for var in vars:
        if abs:
            summed = np.abs(spatial_sum(ds, var, mult_dv=mult_dv, spatial_dim=spatial_dim))
        else:
            summed = spatial_sum(ds, var, mult_dv=mult_dv, spatial_dim=spatial_dim)
        ax.plot(time, summed, label=f'{var} ({ds[var].units})', linewidth=linewidth, markersize=markersize)


    if plot_src:
        ax2 = ax.twinx()
        summed_src = spatial_sum(ds, 'Pe_src', mult_dv=False, spatial_dim=spatial_dim)
        ax2.plot(time, summed_src, label='Pe_src', color='black', linewidth=linewidth, markersize=markersize)
    ax.set_xlabel('Time (ms)')
    if mult_dv:
        ax.set_ylabel(fr'Spatial Sum ($\sum$ vars $\cdot  dV$)')
    else:
        ax.set_ylabel(fr'Spatial Sum ($\sum$ vars)')
        
    # ax.set_title('Spatial Sum Over Time')
    if y_scale is not None:
        ax.set_yscale(f'{y_scale}')
    else:
        ax.set_yscale('linear')

    if title is not None:
        ax.set_title(title, loc = 'right')

    def smart_ncols(n_items):
        if n_items <= 3:
            return n_items
        elif n_items <= 8:
            return 2
        elif n_items <= 12:
            return 3
        else:
            return 4

    ncols = smart_ncols(len(vars))

    fig.legend(ncol=ncols, bbox_to_anchor=(0.5, 1.1), loc='upper right')
    ax.grid(True)
    plt.show()



def source_sink(ds, var, time_range, symlog=True):
    """
    Plots the integrated value of a variable over time for a specified time range.

    Parameters:
    - ds: xarray.Dataset
    - var: str, variable name in the dataset
    - time_range: tuple(int, int), time index range as (start, stop)
    - symlog: bool, whether to use symlog scale on y-axis
    """

    # Initialize lists
    pump = []
    times = []

    # Select the subset using isel and a slice
    ds_subset = ds.isel(t=slice(time_range[0], time_range[1]))

    for t_idx in range(len(ds_subset['t'])):
        t_global_idx = t_idx + time_range[0]
        current_time = (ds['t'].values[t_global_idx] - ds['t'].values[0]) * 1e3  # in ms
        times.append(current_time)

        # Sum over spatial domain, removing guard cells
        pump_tot = np.sum(ds[var].isel(t=t_global_idx).values[1:-1])
        pump.append(pump_tot)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.plot(times, pump, color='blue', label='Target Recycle', linewidth=2, marker='o', markersize=5)

    ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_ylabel(fr"$\sum$ {ds[var].name} ({ds[var].units})", fontsize=16)
    ax.grid(True)

    if symlog:
        ax.set_yscale('symlog')

    else:
        ax.set_yscale('log')

    # ax.legend()
    plt.tight_layout()
    plt.show()

def my_front_tracking_1D(ds, more_fronts = False):
    ds = ds.copy()
    df = pd.DataFrame()
    df.index = range(ds.dims["t"])
    dist = ds["pos"].values
    df["t"] = ds["t"]

    def find_crossing(dist, data, threshold):

        # Find indices where the temperature crosses the threshold
        final_crossing = np.where(np.diff(np.signbit(data - threshold)))[0][-1]

        # Initialize a list to store crossing times
        crossing_times = []

        # Interpolate within each crossing interval to find the exact crossing time
        t1, t2 = dist[final_crossing], dist[final_crossing + 1]
        data1, data2 = data[final_crossing], data[final_crossing + 1]

        # Linear interpolation to find the crossing time
        location = t1 + (threshold - data1) * (t2 - t1) / (data2 - data1)

        return location



    for t in range(ds.dims["t"]):

        timeslice = ds.isel(t=t)
        df.loc[t, "5eV"] = find_crossing(dist, timeslice["Te"].values, 5)

        if more_fronts is True:
            df.loc[t, "Ne_peak"] = find_crossing(dist, timeslice["Ne"].values, timeslice["Ne"].values.max())
            df.loc[t, "Rneon"] = find_crossing(dist, timeslice["Rneon"].values, timeslice["Rneon"].values.max())
            df.loc[t, "iz_peak"] = find_crossing(dist, timeslice["Sd+_iz"].values, timeslice["Sd+_iz"].values.max())
            df.loc[t, "rec_peak"] = find_crossing(dist, timeslice["Sd+_rec"].values, timeslice["Sd+_rec"].values.max())

    # print(df.index)
    # print(df.index.duplicated())

    ds["front_pardist_5eV"] = xr.DataArray(df["5eV"].values, dims=["t"], coords={"t": ds["t"].values})
    ds["front_pardist_5eV"].attrs.update(dict(
        short_name = "5eV front pol. distance from target [m]",
        units = "m",
        origin = "sdtools")
    )

    
    if more_fronts is True:
        ds["front_poldist_Rpeak"] = xr.DataArray(df["Rneon"].values, dims = ["t"])
        ds["front_poldist_Rpeak"].attrs.update(dict(
            short_name = "R peak front pol. distance from target [m]",
            units = "m",
            origin = "sdtools")
        )

        ds["front_poldist_IZpeak"] = xr.DataArray(df["iz_peak"].values, dims = ["t"])
        ds["front_poldist_IZpeak"].attrs.update(dict(
            short_name = "IZ peak front pol. distance from target [m]",
            units = "m",
            origin = "sdtools")
        )

        ds["front_poldist_Nepeak"] = xr.DataArray(df["Ne_peak"].values, dims = ["t"])
        ds["front_poldist_Nepeak"].attrs.update(dict(
            short_name = "Ne peak front pol. distance from target [m]",
            units = "m",
            origin = "sdtools")
        )


    return ds
# ds["front_poldist_5eV"] = xr.DataArray(df["5eV"].values, dims = ["t"])

import numpy as np

def detachment_cloud(ds, time_index = 0, verbose = False, calc_iz_energy = False):
    neutrals = ds['Nd'].isel(t=time_index).values
    cloud_indx = np.where(neutrals > 1e14)[0][0]

    neutral_cloud = neutrals[cloud_indx:-2]
    volume = ds['dv'].values[cloud_indx:-2]

    neutral_number = np.sum(neutral_cloud * volume)

    if verbose:
        print(f'Detachment cloud neutral density (1D): {neutral_cloud[-5]}', r'$m^{-3}$')
        print(f"Detachment cloud neutral number (1D): {neutral_number} #")
        
    if calc_iz_energy:
        assumed_effective_ionisation_energy = 30  # eV
        neutral_ionisation_energy = neutral_number * assumed_effective_ionisation_energy * 1.6e-19
        print(f'Neutral ionisation energy (1D): {neutral_ionisation_energy:.3e} J')
    return neutral_number

# print('before pulse')
# detachment_cloud(cs['power_only'].ds, time_index=0,verbose=True)

# print('peak pulse')
# detachment_cloud(cs['power_only'].ds, time_index=100, verbose=True)



from PIL import Image

def replace_guards(var):
    """
    Replace the points in the guard cells with boundary values.
    """
    # print(var.values)
    var = np.ravel(var.values)[1:-1]  # Strip the edge guard cells

    var[0] = 0.5 * (var[0] + var[1])
    var[-1] = 0.5 * (var[-1] + var[-2])
    
    return var


import matplotlib.pyplot as plt
from cycler import cycler
def spatial_var(ds_in, time, var=[''], second_axis=[], x_log_scale=True, title='None',
                save=False, save_path=None, focus_target=False, y_scale='log', y_scale2='linear',
                first_profile=None, ylim1=None, ylim2=None, close_plot = False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    linewidth = 2
    ds = ds_in.ds.copy().isel(t=time)

    colors = plt.cm.tab10.colors
    color_cycler = cycler(color=colors)
    ax.set_prop_cycle(color_cycler)

    y = ds['y'][1:-1].values
    y_flipped = (y.max() - y)

    for idx, i in enumerate(var):
        color = colors[idx % len(colors)]
        if first_profile is not None:
            ax.plot(y_flipped, first_profile[i], linestyle='--', linewidth=linewidth,
                    alpha=0.5, color=color)
        ax.plot(y_flipped, replace_guards(ds[i]), label=f'{i}', linewidth=linewidth, color=color)

    ax2 = None
    if second_axis:
        ax2 = ax.twinx()
        ax2.set_prop_cycle(cycler(color=colors[len(var):]))
        for idx, j in enumerate(second_axis):
            color = colors[(len(var) + idx) % len(colors)]
            if first_profile is not None:
                ax2.plot(y_flipped, first_profile[j], linestyle='--', linewidth=linewidth,
                         alpha=0.5, color=color)
            ax2.plot(y_flipped, replace_guards(ds[j]), label=f'{j}', linewidth=linewidth, color=color)
        ax2.set_ylabel(f"{ds[second_axis[0]].units}", fontsize=20)
        ax2.set_yscale(y_scale2)
        if ylim2:
            ax2.set_ylim(*ylim2)

    ax.set_xscale('log' if x_log_scale else 'linear')
    if focus_target:
        ax.set_xlim(0, 5)

    ax.set_yscale(y_scale)
    ax.set_xlabel('Distance to target (m)', fontsize=20)
    ax.set_ylabel(f'{ds[var[0]].units}', fontsize=20)

    if ylim1:
        ax.set_ylim(*ylim1)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    fig.legend(handles1 + handles2, labels1 + labels2, ncol=4, loc='upper right')

    fig.suptitle(f"{(ds_in.ds['t'].values[time] - ds_in.ds['t'].values[0]) * 1e3 : 6f} ms \n time_index = {time}",
                 fontsize=20, x=0.0, horizontalalignment='left')

    if save:
        plt.savefig(save_path, bbox_inches='tight')

    if close_plot:
        plt.close()

from tqdm import tqdm
import os
import imageio
import numpy as np
from PIL import Image

def spatial_var_mp4(ds_in, start_time, end_time, var=[''], second_axis=[], fps=5,
                    mp4_filename="spatial_var.mp4", x_log_scale=True,
                    focus_target=False, y_scale='log', y_scale2='linear',
                    ylim1=None, ylim2=None):
    output_dir = "./frames"
    os.makedirs(output_dir, exist_ok=True)

    first_ds = ds_in.ds.copy().isel(t=start_time)
    first_profile = {i: replace_guards(first_ds[i]) for i in var + second_axis}

    # Frame generation with progress bar
    frame_paths = []
    total_frames = end_time - start_time
    
    print("Generating frames...")
    for t in tqdm(range(start_time, end_time), desc="Creating frames", unit="frame"):
        frame_path = os.path.join(output_dir, f"frame_{t:03d}.png")
        spatial_var(ds_in, time=t, var=var, second_axis=second_axis, save=True,
                    save_path=frame_path, x_log_scale=x_log_scale,
                    y_scale=y_scale, y_scale2=y_scale2,
                    focus_target=focus_target,
                    first_profile=first_profile, ylim1=ylim1, ylim2=ylim2, close_plot=True)
        frame_paths.append(frame_path)

    # Video creation with progress bar
    first_frame = imageio.imread(frame_paths[0])
    target_size = (first_frame.shape[1], first_frame.shape[0])

    print("Creating MP4...")
    writer = imageio.get_writer(mp4_filename, fps=fps, codec='libx264', format='ffmpeg')
    
    for path in tqdm(frame_paths, desc="Writing video", unit="frame"):
        frame = imageio.imread(path)
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = np.array(Image.fromarray(frame).resize(target_size, Image.BICUBIC))
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        writer.append_data(frame)
    
    writer.close()
    print(f"Video saved as: {mp4_filename}")
    print(f"Total frames processed: {len(frame_paths)}")



if __name__ == "__main__":
    print('hello')