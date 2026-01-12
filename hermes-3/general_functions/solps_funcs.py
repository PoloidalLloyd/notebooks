from boututils.datafile import DataFile
from boutdata.collect import collect
import pandas as pd
import numpy as np
import scipy as cp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import os, sys, pathlib
import platform
import traceback
import xarray as xr
import xbout
from pathlib import Path
import xhermes as xh

sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/sdtools"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/transients"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/general_functions"))


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
from hermes3.neutral_transport import *

# from code_comparison.code_comparison import *
from code_comparison.solps_pp import *
from code_comparison.solps_variables import *
# from code_comparison.viewer_2d import *

# from hermes3.balance1d import *

# plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
linewidth = 3
markersize = 15



# plt.style.use('ggplot')
mpl.style.use('default')
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.grid'] = True
mpl.rcParams.update({'font.size': 16})





def plot_radial_profiles(case, param_2d = 'Te', vmin = 0, vmax = 100, region='omp', Te_threshold=5, sepadd=2, 
                        keep_geometry=True, verbose=True, zoom = False, grid_only = False, 
                        cfr_only=False, poloidal_offset=0):
    """
    Plots a summary of radial plasma profiles and their 2D location for a given region.

    Parameters:
        case: SOLPS case object
        region: Region string (e.g., 'omp', 'outer_lower_detachment_front', etc.)
        Te_threshold: Te threshold for detachment front (if needed)
        sepadd: SOL ring to search for front
        keep_geometry: If True, retain R/Z data in DataFrame
        verbose: Print more information from .get_1d_radial_data
        cfr_only: If True, only plot profile points where dist >= 0
        poloidal_offset: Integer number of grid cells to offset from the base region location.
                        Positive values move "downward" (toward lower target), negative move "upward".
                        When non-zero, interpolation to Z=0 is disabled. (default: 0)
    """
    # Create figure with custom gridspec layout
    fig = plt.figure(figsize=(30, 12))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])

    # Create axes - 4 plots on left (2x2), 1 tall plot on right
    ax_temp = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_press = fig.add_subplot(gs[1, 0])
    ax_dens = fig.add_subplot(gs[1, 1])
    ax_2d = fig.add_subplot(gs[:, 2])  # Spans both rows

    radial_data = case.get_1d_radial_data(
        params=['Te', 'Td+', 'Ta', 'Td', 'Tm', 'Tn', 'Pe', 'Pd+', 'Pa', 'Pn', 'Pm',
                'Ne', 'Nn', 'Nm', 'Na', 'fhex_total',
                'fhix_total', 'fhey_total', 'hy', 'R', 'Z', 'fhx_density', 'fhy_density', 'fhtx_density', 'fhty_density'],
        region=region,
        Te_threshold=Te_threshold,
        sepadd=sepadd,
        keep_geometry=keep_geometry,
        verbose=verbose,
        poloidal_offset=poloidal_offset
    )
    
    # Update title to show offset if non-zero
    if poloidal_offset != 0:
        fig.suptitle(f'{region} (poloidal_offset={poloidal_offset})')
    else:
        fig.suptitle(f'{region}')

    # Apply filter for dist >= 0 if requested by cfr_only
    if cfr_only:
        radial_data_profile = radial_data[radial_data['dist'] >= 0].copy()
    else:
        radial_data_profile = radial_data

    # Plot 1: Temperature profiles
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Te'], 'o-', label='Te', color='blue')
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Td+'], 'o-', label='Td+', color='red')
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Ta'], 'o-', label='Ta', color='green')
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Td'], 'o-', label='Td', color='orange')
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Tm'], 'o-', label='Tm', color='purple')
    ax_temp.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Tn'], 'o-', label='Tn', color='brown')
    ax_temp.axvline(0, color='k', linestyle='--', alpha=0.5, label='Separatrix')
    ax_temp.set_xlabel('Distance from separatrix [mm]')
    ax_temp.set_ylabel('Temperature [eV]')
    ax_temp.legend()
    ax_temp.grid(True, alpha=0.3)

    # Plot 2: Parallel heat flux
    ax_heat.plot(radial_data_profile['dist'] * 1000, radial_data_profile['fhex_total']/radial_data_profile['apar'], 'o-', label='fhex_total', color='blue')
    ax_heat.plot(radial_data_profile['dist'] * 1000, radial_data_profile['fhix_total']/radial_data_profile['apar'], 'o-', label='fhix_total', color='red')
    ax_heat.plot(radial_data_profile['dist'] * 1000, radial_data_profile['fhtx_density'], 'o-', label='fhtx_density', color='green')
    # ax_heat.plot(radial_data_profile['dist'] * 1000, radial_data_profile['fhty_density'], 'o-', label='fhty_density', color='purple')
    ax_heat.axvline(0, color='k', linestyle='--', alpha=0.5, label='Separatrix')
    ax_heat.set_xlabel('Distance from separatrix [mm]')
    ax_heat.set_ylabel('Parallel heat flux [W/m]')
    ax_heat.legend()
    ax_heat.grid(True, alpha=0.3)

    # Plot 3: Pressure profiles
    ax_press.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Pe'], 'o-', label='Pe', color='blue')
    ax_press.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Pd+'], 'o-', label='Pd+', color='red')
    ax_press.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Pa'], 'o-', label='Pa', color='green')
    ax_press.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Pm'], 'o-', label='Pm', color='orange')
    ax_press.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Pn'], 'o-', label='Pn', color='purple')
    ax_press.axvline(0, color='k', linestyle='--', alpha=0.5, label='Separatrix')
    ax_press.set_xlabel('Distance from separatrix [mm]')
    ax_press.set_ylabel('Pressure [Pa]')
    ax_press.legend()
    ax_press.grid(True, alpha=0.3)

    # Plot 4: Density profiles
    ax_dens.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Ne'], 'o-', label='Ne', color='blue')
    ax_dens.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Nn'], 'o-', label='Nn', color='red')
    ax_dens.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Nm'], 'o-', label='Nm', color='green')
    ax_dens.plot(radial_data_profile['dist'] * 1000, radial_data_profile['Na'], 'o-', label='Na', color='orange')
    ax_dens.axvline(0, color='k', linestyle='--', alpha=0.5, label='Separatrix')
    ax_dens.set_xlabel('Distance from separatrix [mm]')
    ax_dens.set_ylabel(r'Density [m$^{-3}$]')
    ax_dens.legend()
    ax_dens.grid(True, alpha=0.3)

    # Plot 5: 2D cross-section with radial cut location
    vmin = vmin
    vmax = vmax
    case.plot_2d(param_2d, ax=ax_2d, logscale=True, cbar=True, vmin=vmin, vmax=vmax, grid_only = grid_only)

    # Plot the radial cut location -- ONLY include dist >=0 if cfr_only is True
    if cfr_only:
        R_cut = radial_data.loc[radial_data['dist'] >= 0, 'R']
        Z_cut = radial_data.loc[radial_data['dist'] >= 0, 'Z']
    else:
        R_cut = radial_data['R']
        Z_cut = radial_data['Z']
    ax_2d.plot(R_cut, Z_cut, linestyle = '--', color = 'black', linewidth=3, label='Radial cut')

    # Mark the separatrix point on the cut (looks for sep==1 in the FULL radial_data, as usual)
    sep_idx = radial_data[radial_data['sep']==1].index[0]
    det_R = radial_data.loc[sep_idx, 'R']
    det_Z = radial_data.loc[sep_idx, 'Z']
    ax_2d.plot(det_R, det_Z, 'wo', markersize=8, markeredgecolor='black',
               markeredgewidth=1.5, label='Separatrix')

    ax_2d.legend()

    # Set y-limits based on region
    if any([k in region for k in ['lower', 'omp', 'imp']]):
        if zoom:
            ax_2d.set_xlim(np.min(det_R) - 0.4, np.max(det_R) + 0.4)
            ax_2d.set_ylim(np.min(det_Z) - 0.4, np.max(det_Z) + 0.4)    
        else:
            ax_2d.set_ylim(-2.2, 0.2)
    else:
        if zoom:
            ax_2d.set_xlim(np.min(det_R) - 0.4, np.max(det_R) + 0.4)
            ax_2d.set_ylim(np.min(det_Z) - 0.4, np.max(det_Z) + 0.4)    
        else:
            ax_2d.set_xlim(0, 1)
            ax_2d.set_ylim(-0.2, 2.2)

    plt.tight_layout()
    plt.show()


from scipy.optimize import curve_fit
from scipy.special import erfc
def get_lambda_q(case, poloidal_offset, map_to = 'omp',map_offset = 0, range=(0.0, None), plot = False):

    # get heat flux at peak parallel power (normally just above the X-point)
    fline_peaks = case.get_1d_radial_data(
        params = ['fhtx', 'R'],
        region = 'omp',
        sepadd = 1,
        poloidal_offset = poloidal_offset,
        guards = False
    )

    fline_R = case.get_1d_poloidal_data(
        params = ['R'],
        region = 'outer_lower',
        sepadd = 0,
        guards = False
    )

    # get radial dist at region we map the parallel power to
    fline_map = case.get_1d_radial_data(
        params = [],
        #region = 'omp',
        region = map_to,
        poloidal_offset = map_offset,
        sepadd = 0,
        guards = False
    )

    # print(fline_R['R'].values[poloidal_offset])
    # print(fline_R['R'].values[-1])

    # Build arrays (robust for non-monotonic dist in x-point regions)
    omp_dist = np.asarray(fline_map['dist'])
    q_parallel = np.asarray(
        (fline_peaks['fhtx'] / fline_peaks['apar'])
        * (fline_R['R'].values[poloidal_offset] / fline_R['R'].values[-1])
    )

    # Defensive: ensure both arrays align by radial index
    n = min(omp_dist.size, q_parallel.size)
    omp_dist = omp_dist[:n]
    q_parallel = q_parallel[:n]

    # Apply dist window using boolean masking (don’t assume dist is sorted)
    lo, hi = range
    mask = np.isfinite(omp_dist) & np.isfinite(q_parallel)
    if lo is not None:
        mask &= (omp_dist >= lo)
    if hi is not None:
        mask &= (omp_dist <= hi)

    omp_dist = omp_dist[mask]
    q_parallel = q_parallel[mask]

    # Sort for nicer plots + stable fitting
    order = np.argsort(omp_dist)
    omp_dist = omp_dist[order]
    q_parallel = q_parallel[order]

    if omp_dist.size < 3:
        raise ValueError(
            f"Not enough points after filtering to fit. map_to='{map_to}', map_offset={map_offset}, range={range}"
        )

    # fig,ax = plt.subplots()
    # ax.plot(omp_dist, q_parallel)
    # plt.show()


    try:
        def fit_exp(r,a,lmq):
            return a*np.exp(-r/lmq)

        p0_exp = [q_parallel[0], 0.006]
        popt_exp, pcov_exp = curve_fit(fit_exp, omp_dist, q_parallel, p0=p0_exp)

        lambda_q = popt_exp[1]
        hwhm = lambda_q * np.log(2)

        print(f'Exponential fit:')
        print(f'lambda_q = {lambda_q*1000:.3f} mm')
        print(f'HWHM = {hwhm*1000:.3f} mm')
        print(f'amplitude = {popt_exp[0]:.2e} W/m^2')
        exponential_success = True
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("Fitting failed. Returning None for lambda_q.")
        exponential_success = False

    # try:
    #     def gaussian_fwhm(r , a, r0 , sigma):
    #         return a*np.exp(-(r-r0)**2/(2*sigma**2))
   
    #     # Better initial guess: peak near separatrix, small sigma similar to lambda_q
    #     p0_gauss = [q_parallel.iloc[0], 0.0, 0.003]  # peak at r=0, small width
        
    #     # Add bounds to keep fit physically reasonable
    #     bounds = (
    #         [0, -0.001, 0.0001],  # lower bounds: positive amplitude, r0 near 0, small sigma
    #         [np.inf, 0.005, 0.02]  # upper bounds: allow r0 to move slightly, reasonable sigma
    #     )
        
    #     popt_gauss, pcov_gauss = curve_fit(
    #         gaussian_fwhm, omp_dist, q_parallel, 
    #         p0=p0_gauss,
    #         bounds=bounds,
    #         maxfev=5000
    #     )
    #     print(f'Gaussian fit:')
    #     print(f'Peak amplitude = {popt_gauss[0]}')
    #     print(f'Peak position = {popt_gauss[1]}')
    #     print(f'FWHM = {popt_gauss[2]}')
    #     gaussian_success = True
    # except RuntimeError as e:
    #     print(f"Runtime error: {e}")
    #     print("Fitting failed. Returning None for lambda_q.")
    #     gaussian_success = False


    if plot:
        fig, ax = plt.subplots(figsize = (10, 8))
        ax.plot(omp_dist*1000, q_parallel*1e-6, label='SOLPS', color = 'blue', marker = 'o')
        ax.axvline(0, color = 'black', linestyle = '--', label = 'Separatrix')
        if exponential_success: 
            ax.plot(omp_dist*1000, fit_exp(omp_dist, *popt_exp)*1e-6, label='Exponential fit', color = 'black', linestyle = ':')
            ax.axvspan(omp_dist[0]*1000, popt_exp[1]*1000, color='red', alpha=0.3, label=rf'$\lambda_q$ = {lambda_q*1000:.3f} mm')
        # if gaussian_success:
        #     ax.plot(omp_dist, gaussian_fwhm(omp_dist, *popt_gauss), label='Gaussian fit')
        ax.legend()
        ax.set_xlabel(r'r-r$_{sep}$' + f' at {map_to} (mm)')
        ax.set_ylabel(r'Parallel heat flux (MW/m$^2$)')
        ax.set_title(f'Peak parallel heat flux mapped to {map_to} \n (offset = {map_offset})')
        plt.show()

    return lambda_q, hwhm, popt_exp[0]





if __name__ == "__main__":
    # Example usage:
    # plot_radial_profiles(case, region='outer_lower_detachment_front', zoom = False, param_2d = 'Te', grid_only = False)
    plot_radial_profiles(case, region='omp', zoom = False, param_2d = 'Te', grid_only = False, cfr_only = True)
    plot_radial_profiles(case, region='outer_lower_xpoint', zoom = False, param_2d = 'Te', grid_only = False, cfr_only = True)
    plot_radial_profiles(case, region='outer_lower_detachment_front', zoom = False, param_2d = 'Te', grid_only = False, cfr_only = True)



    # Example usage:    
    # Exponential fit
    result = get_lambda_q(case, region='omp', method='exponential', range=(0.0, None))
    lambda_q = result['lambda_q']

    # Eich fit
    result = get_lambda_q(case, region='omp', method='eich', range=(0.0, None))
    lambda_q = result['lambda_q']

    # Gaussian FWHM
    result = get_lambda_q(case, region='omp', method='gaussian', range=(0.0, None))
    fwhm = result['fwhm']