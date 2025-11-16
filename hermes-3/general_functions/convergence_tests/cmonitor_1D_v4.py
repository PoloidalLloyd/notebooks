#!/usr/bin/env python
import matplotlib.pyplot as plt
import os, sys, pathlib
import numpy as np
import xarray as xr
import xhermes as xh
import matplotlib.animation as animation
import time

sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/sdtools"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/general_functions"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/general_functions/source_functions.py"))
from convergence_functions import *
from plotting_functions import *
from source_functions import *
from matplotlib.ticker import LogFormatter
from gpu_accelerated_animation import create_animation_ultra_fast_gpu, quick_gpu_animation


def main(directory_path):
    editors = ['code', 'cursor']
    if directory_path == ".":
        casename = os.getcwd()
    else:
        casename = directory_path
    print(f"Reading {casename}")

    # Establish the data directory (absolute path, resolve symlinks, etc.)
    data_dir = os.path.abspath(casename)

    # For single file: use the parent directory; for directory: use as is.
    if os.path.isfile(data_dir):
        output_dir = os.path.dirname(data_dir)
    else:
        output_dir = data_dir

    # Ensure trailing slash is absent (normalize path)
    output_dir = os.path.normpath(output_dir)

    while True:
        try:
            ds = xh.open(casename)  # Attempt to open the dataset

            # Change to output directory so plots are saved there
            original_dir = os.getcwd()
            os.chdir(output_dir)

            # Generate time history plot
            plot_time_history(
                ds, 
                variables=['Te', 'Td+', 'Ne', 'Nd', 'Sd+_src', 'Pe_src'], 
                track_detachment_front=True, 
                save=True
            )

            # Generate profiles plot
            plot_profiles(
                ds, 
                variables=['Te', 'Td+', 'Ne', 'Nd'], 
                data_label='Simulation', 
                save=True
            )

            try:
                pi_feedback_source(
                    ds, 
                    plot=False, 
                    time_slices=10, 
                    save=True,
                    filename='pi_feedback_source.png'
                )
            except Exception:
                print('No pi feedback source found')

            # Generate profiles animation
            create_animation_ultra_fast_gpu(
                ds,
                variables=['Te', 'Td+', 'Ne', 'Nd'],
                max_frames=15, 
                quality='high',  # 120 DPI instead of 50
                filename='profiles_animation.mp4'
            )

            # Restore original directory
            os.chdir(original_dir)

            break  # Exit the loop once the reading and processing is successful

        except (RuntimeError, OSError, ValueError) as e:
            # Restore original directory on error
            try:
                os.chdir(original_dir)
            except:
                pass
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