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

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions/"))
from convergence_functions import *

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

def check_convergence(directory_path):
    """
    Checks convergence for Hermes-3 by plotting profiles and time histories,
    and saves the generated figures in the current working directory.

    Parameters:
    directory_path (str): Path to the parent directory containing Hermes-3 dmp files.
    """

    
    if directory_path == ".":
        casename = os.path.basename(os.getcwd())
    else:
        casename = os.path.basename(directory_path)
    print(f"Reading {casename}")
    print("Calculating...", end="")

    # Open the dataset using xhermes
    dataset = xh.open(casename)

    # Get the current working directory
    save_dir = os.getcwd()

    # Plot and save the latest profiles figure
    plot_latest_profiles(dataset)
    latest_profiles_fig_path = os.path.join(save_dir, 'latest_profiles.png')
    plt.savefig(latest_profiles_fig_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    # Plot and save the time history figure
    plot_time_history(dataset)
    time_history_fig_path = os.path.join(save_dir, 'time_history.png')
    plt.savefig(time_history_fig_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    print(f"Convergence plots saved to:\n- {latest_profiles_fig_path}\n- {time_history_fig_path}")
