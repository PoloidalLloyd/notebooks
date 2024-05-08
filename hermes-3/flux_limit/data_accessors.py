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


if __name__ == '__main__':
    # Load the dataset
    path = '/shared/storage/plasmahwdisks/data/jlb647/simulation_data/flux_limiter_detachment/2023-12-15_wigram_reference_detached/500MW_5x10(19)'
    ds = load_dataset_1D(path)

    # Plot the dataset
    ds['Te'].plot()
    # plt.show()