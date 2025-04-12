import matplotlib.pyplot as plt
import os, sys, pathlib
import numpy as np
import xarray as xr
import xhermes as xh
import matplotlib.animation as animation
import time

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

    while True:
        try:
            ds = xh.open(casename)  # Attempt to open the dataset

            # Generate time history plot
            plot_time_history(ds, variables=['Te', 'Td+', 'Ne', 'Nd', 'Sd+_src', 'Pe_src'], track_detachment_front=True, save=True)
            
            # Generate profiles plot
            plot_profiles(ds, variables=['Te', 'Td+', 'Ne', 'Nd'], data_label='Simulation', save=True)

            try:
                pi_feedback_source(ds, plot=True, time_slices=10, save=True)
            except:
                print('No pi feedback source found')

            # Generate profiles animation
            plot_profiles_animation(ds, variables=['Te', 'Td+', 'Td', 'Ne', 'Nd','Pe','Pd+', 'Pd', 'NVd+', 'NVd'], data_label='Simulation', filename='profiles_animation.gif')

            
            import imageio

            # Read the GIF file
            gif_reader = imageio.get_reader('profiles_animation.gif')

            # Write the MP4 file
            with imageio.get_writer('profiles_animation.mp4', fps=5) as writer:
                for frame in gif_reader:
                    writer.append_data(frame)


            break  # Exit the loop once the reading and processing is successful


        except RuntimeError or OSError or ValueError as e:
            # Check if the error is related to reading the NetCDF file (HDF error)
            print(str(e))
            print("Error: Failed to read the NetCDF file. It might be in use by the simulation.")
            print("Retrying...")

            # Sleep for a short time before trying again
            time.sleep(0.5)  # Adjust the sleep time if needed


if __name__ == "__main__":
    # Get the simulation data path from the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python plot_convergence.py <simulation_data_path>")
    else:
        simulation_data_path = sys.argv[1]
        main(simulation_data_path)