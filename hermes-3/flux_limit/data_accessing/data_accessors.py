import matplotlib.pyplot as pltimport
import xhermes as xh
from boutdata.data import BoutData
from boutdata import collect
import matplotlib.pyplot as plt
import glob     
import re
import numpy as np
import pandas as pd
import xarray as xr
import time as time

def replace_guards(var):
    """
	This in-place replaces the points in the guard cells with the points on the boundary
    
    """
    # Strip the edge guard cells
    var = var[1:-1]

    var[0] = 0.5*(var[0] + var[1])
    var[-1] = 0.5*(var[-1] + var[-2])
    return var

def load_dataset_1D(path, last_time_step = True):
    """
    Load the dataset from a given path
    """
    # Load the dataset
    if last_time_step:
        ds = xh.open(path).isel(t=-1)
    # Replace the guard cells
    else:
        ds = xh.open(path)

    for i in (ds.data_vars):
        print(i ,type(ds[f'{i}']))
        try:    
            ds[i].values = replace_guards(np.ravel(ds[i].values))
                # print('it worked!')
        except:   
            print('it did not work')

    return ds

def load_fld_last_time_slice(base_dir, alpha_values, neon_values):
    data = {}
    for alpha in alpha_values:
        for neon in neon_values:
            # Construct the path to the specific simulation
            sim_path = os.path.join(base_dir, f'alpha_{alpha}', f'neon_{neon}')
            try:
                print('loading data for alpha={}, neon={}'.format(alpha, neon))
                # Load the latest timestep (t=-1) using Mike's load.case_1D method
                ds = Load.case_1D(sim_path).ds.isel(t=-1)
                data[(alpha, neon)] = ds
            except Exception as e:
                print(f"Failed to load data for alpha={alpha}, neon={neon}: {e}")
    return data


def save_last_time_slice(base_dir, alpha_values, neon_values):
    for alpha in alpha_values:
        for neon in neon_values:
            # Construct the path to the specific simulation
            sim_path = os.path.join(base_dir, f'alpha_{alpha}', f'neon_{neon}')
            try:
                print(f'Loading data for alpha={alpha}, neon={neon}')
                # Load the latest timestep (t=-1) using BOUT's load.case_1D method
                ds = Load.case_1D(sim_path).ds.isel(t=-1)
                
                # Create a daughter directory for the last time slice
                save_dir = os.path.join(base_dir, 'last_time_slice', f'alpha_{alpha}', f'neon_{neon}')
                os.makedirs(save_dir, exist_ok=True)
                
                # Save the last time slice as BOUT.dmp file in the daughter directory
                save_path = os.path.join(save_dir, 'BOUT.dmp.1.nc')
                ds.to_netcdf(save_path)
                
                print(f'Saved last time slice to {save_path}')
            except Exception as e:
                print(f"Failed to process data for alpha={alpha}, neon={neon}: {e}")




if __name__ == '__main__':
    # Load the dataset
    path = '/shared/storage/plasmahwdisks/data/jlb647/simulation_data/flux_limiter_detachment/2023-12-15_wigram_reference_detached/500MW_5x10(19)'
    ds = load_dataset_1D(path)

    # Plot the dataset
    ds['Te'].plot()
    # plt.show()