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


def read_last_time_step(sim_dir, unnormalise=True):
    """
    Read the last time step from a BOUT++ simulation directory.
    
    Parameters
    ----------
    sim_dir : str
        Path to the BOUT++ simulation directory.
        
    Returns
    -------
    int
        The last time step.
    """
    # Get the last time step
    ds = xh.open(sim_dir)

    # Get the last time step
    last_time_step = ds['t'][-1].values

    if unnormalise:
        # un-normalise
        last_time_step = last_time_step * ds['t'].attrs['conversion']
        return last_time_step
    else:
        return last_time_step



