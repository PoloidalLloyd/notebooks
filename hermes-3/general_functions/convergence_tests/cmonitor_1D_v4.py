import matplotlib.pyplot as plt
import os, sys, pathlib
import numpy as np
import xarray as xr
import xhermes as xh
import matplotlib.animation as animation

sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/sdtools"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions"))
sys.path.append(os.path.join(r"/users/jlb647/scratch/simulation_program/hermes-3_sim/analysis/my_notebooks/notebooks/hermes-3/general_functions/source_functions.py"))
from convergence_functions import *
from plotting_functions import *
from source_functions import *
from matplotlib.ticker import LogFormatter


def main(directory_path):
    if directory_path == ".":
        casename = os.getcwd()
    else:
        casename = directory_path
    print(f"Reading {casename}")

    ds = xh.open(casename)

    # Generate time history plot
    plot_time_history(ds, variables=['Te','Td+', 'Ne', 'Nd', 'SNd+'], track_detachment_front=True, save = True)
    
    # Generate profiles plot
    plot_profiles(ds, variables=['Te','Td+', 'Ne', 'Nd'], data_label='Simulation', save=True)

    try:
        pi_feedback_source(ds, plot = True, time_slices = 10, save = True)
    except:
        print('No pi feedback source found')

    # Generate profiles animation
    # plot_profiles_animation(ds, variables=['Te','Td+', 'Ne', 'Nd'], data_label='Simulation', filename='profiles_animation.gif')

if __name__ == "__main__":

    # Get the simulation data path from the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python plot_convergence.py <simulation_data_path>")
    else:
        simulation_data_path = sys.argv[1]
        main(simulation_data_path)
