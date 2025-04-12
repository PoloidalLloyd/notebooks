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
import re

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





def ttod_inputs(case_path, grid_path, region=('outer_lower'), sep_add = [1], verbose = False, plot_geometry = False):


    # Load the case
    db = CaseDB(case_dir = case_path, grid_dir= grid_path)

    case_dir = os.path.basename(case_path)

    case = db.load_case_2D(case_dir, use_squash = True, verbose = True)

    # get flux tube of interest with params (too many currently but who cares!)
    params = ['R', 'Bxy', 'efd+_cond_ylow', 'efd+_kin_ylow', 'efd+_tot_xlow', 'efd+_tot_ylow', 
          'efe_cond_ylow' , 'efe_kin_ylow', 'efe_tot_ylow', 'kappa_par_e', 'Te', 
          'efd+_tot_xlow','efe_tot_xlow', 'J', 'g_22', 'dx', 'dy','dz', 'Td+', 'dr', 'dl', 'SNd+', 'SNd','pfd+_tot_xlow']
    

    for i in sep_add:
        profile = get_1d_poloidal_data(case.ds.isel(t=-1), params = params, region = ('outer_lower'), sepadd = i)
        # Get geometry
        x_point_index = np.argmin(profile['R'])
        target_index = np.argmax(profile['Spar'])

        sys_len = profile['Spar'][target_index]
        sys_xpt_len = profile['Spar'][x_point_index]


        # fit flux expansion

        # Get upstream density
        sys_density = profile['SNd+'][0]

        # Get heat flux at xpoint
        ## determine area of flux tube
        dx_conv = case.ds['dx'].conversion
        da = profile['dx']*profile['dz']*dx_conv*profile['J']/np.sqrt(profile['g_22'])

        sys_xpt_heat_flux_e = profile['efe_tot_ylow'][x_point_index]/da[x_point_index]
        sys_xpt_heat_flux_i = profile['efd+_tot_ylow'][x_point_index]/da[x_point_index]

        # get impurity type and fraction
        # Regex to match 'fixed_fraction_*' and capture species name
        fixed_fraction_pattern = r"fixed_fraction_(\S+)"
        
        # Regex to match 'fraction = **' and capture the fraction value
        fraction_pattern = r"fraction = (\d+\.?\d*)"
        
        # Open the file and read all lines
        with open(case_path + '/BOUT.inp', 'r') as file:
            lines = file.readlines()

        # Iterate through lines to find the matching patterns
        species = None
        fraction_value = None

        try:
            for i, line in enumerate(lines):
                # Search for 'fixed_fraction_*' pattern
                match_fixed_fraction = re.search(fixed_fraction_pattern, line)
                if match_fixed_fraction:
                    # If found, extract the species name (e.g., 'carbon')
                    species = match_fixed_fraction.group(1)
                    
                    # Now look for 'fraction = **' in the next lines
                    for j in range(i + 1, len(lines)):  # Start from the next line
                        match_fraction = re.search(fraction_pattern, lines[j])
                        if match_fraction:
                            # If found, extract the fraction value (e.g., '0.008')
                                fraction_value = match_fraction.group(1)
                                break
                # else:
                #     species = 'None_found'
                #     fraction_value = 'None_found'
        except:
            print('Error parsing species and fraction value')
            species = 'None_found'
            fraction_value = 'None_found'
            


        # Plot geometry
        if plot_geometry:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(profile['R'], profile['Z'], label='Flux tube')
            ax.legend()
            plt.show()

        if verbose:
            print('-----------------------------------------------------')
            print('sys_len =', sys_len)
            print('sys_xpt_len =', sys_xpt_len)
            print('sys_density =', sys_density)
            print('sys_xpt_heat_flux_e =', sys_xpt_heat_flux_e)
            print('sys_xpt_heat_flux_i =', sys_xpt_heat_flux_i)
            print(f"Species: {species}, Fraction: {fraction_value}")

            print('-----------------------------------------------------')



if __name__ == '__main__':


    # do something?
    ttod_inputs('/users/jlb647/scratch/simulation_program/hermes-3_sim/simulation_dir/2025-01_STEP_1D-2D_comparison/Mike_2D_cases/m4ab-tune_albedo_new_branch', 
                '/users/jlb647/scratch/simulation_program/hermes-3_sim/sdtool_load_test/grids', verbose = True, plot_geometry = True)