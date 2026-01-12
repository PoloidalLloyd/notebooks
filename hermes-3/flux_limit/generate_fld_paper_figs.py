#!/usr/bin/env python3
"""
Script to generate figures for the flux limiter detachment paper.
Converts the Jupyter notebook 2025-12_fld_paper_figs.ipynb into a standalone Python script.
"""

import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import scipy.integrate

# Add paths for custom modules
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/sdtools"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/general_functions"))
sys.path.append(os.path.join(r"/Users/lloyd/Documents/hermes_dir/analysis/notebooks/hermes-3/hermes-3/general_functions"))

from heatflux_functions import *
from plotting_functions import *
from convergence_functions import *

from hermes3.case_db import *
from hermes3.casedeck import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
from hermes3.fluxes import *
from hermes3.selectors import *

# Set matplotlib style and parameters
plt.style.use('default')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 16})

linewidth = 3
markersize = 10

# Global variables
alpha_vals = [-1, 0.2, 0.06, 'snb']
neon_vals = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
parent_dir = '/Users/lloyd/Documents/hermes_dir/flux_limiter_detachment/Long_runs'
cs = dict()


def replace_guards(var):
    """
    This in-place replaces the points in the guard cells with the points on the boundary
    """
    # Strip the edge guard cells
    var = var[1:-1]
    var[0] = 0.5*(var[0] + var[1])
    var[-1] = 0.5*(var[-1] + var[-2])
    return var


def spitzer_q(dataframe):
    """Calculate Spitzer-Harm heat flux"""
    x = dataframe['y']
    Te = dataframe['Te']
    Ti = dataframe['Td+']
    Ne = dataframe['Ne']
    Ni = dataframe['Nd+']
    kappa_e = dataframe['kappa_par_e']
    kappa_i = dataframe['kappa_par_d+']
    
    grad_T = np.gradient(Te, x)
    q = -kappa_e * grad_T
    return q


def divq_integrate(dataframe, snb_int=False):
    """
    Calculate the total heat flux from the divergence of the Spitzer-Harm fluxes.
    If snb_int == True outputs integral of divq_snb, otherwise outputs integral of divq_sh.
    """
    x = np.ravel(dataframe['y'].values)[1:-1]
    Te = np.ravel(dataframe['Te'].values)
    
    div_q_snb = replace_guards(np.ravel(dataframe['Div_Q_SNB'].values))
    
    print('x =', len(x))
    print('div_q_snb =', len(div_q_snb))
    
    q_snb = scipy.integrate.cumulative_trapezoid(div_q_snb, x, initial=0)
    return q_snb


def neon_adas_curve(Te):
    """Calculate neon ADAS radiation curve"""
    logT = np.log(Te)
    log_out = np.zeros_like(Te)
    
    # Coefficients for the polynomial part
    coefficients = np.array([
        -8.21475117e+01, 1.28929854e+01, -4.74266289e+01,
        7.45222324e+01, -5.75710722e+01, 2.57375965e+01,
        -7.12758563e+00, 1.24287546e+00, -1.32943407e-01,
        7.97368445e-03, -2.05487897e-04
    ])
    
    # Valid temperature range for the polynomial
    valid_mask = (Te >= 2) & (Te <= 1000)
    below_mask = Te < 2
    above_mask = Te > 1000
    
    # Apply the polynomial coefficients
    for i, coeff in enumerate(coefficients):
        log_out[valid_mask] += coeff * logT[valid_mask] ** i
    
    # Apply the exponential to the valid range
    log_out[valid_mask] = np.exp(log_out[valid_mask])
    
    # Assign the constant values outside the valid temperature range
    log_out[below_mask] = 6.35304113e-36
    log_out[above_mask] = 1.17894628e-32
    
    return log_out


def load_data():
    """Load all simulation cases"""
    print("Loading simulation data...")
    for alpha in alpha_vals:
        for neon in neon_vals:
            name = f"alpha_{alpha}_neon_{neon}"
            print(f"Loading {name}")
            cs[name] = Load.case_1D(f'{parent_dir}/alpha_{alpha}/neon_{neon}', 
                                     guard_replace=False, use_squash=True)
    print("Data loading complete.")


def get_alpha_style(alpha_val):
    """Get linestyle, color, and label for a given alpha value"""
    if alpha_val == -1:
        return '--', 'red', 'SH'
    elif alpha_val == 0.2:
        return '-', 'blue', r'$\alpha = 0.2$'
    elif alpha_val == 0.06:
        return '-', 'black', r'$\alpha = 0.06$'
    elif alpha_val == 'snb':
        return ':', 'magenta', 'SNB'
    else:
        return '-', 'gray', f'alpha = {alpha_val}'


def figure_2_8_profiles(neon_value='0.0', inset_xscale='linear', output_dir='figures'):
    """
    Generate Figure 2/8: Profiles Te, q_parallel_e, Ne, Nd with insets
    """
    print(f"Generating Figure 2/8: Profiles for neon={neon_value}")
    
    params = ['Te', 'q_cond', 'Ne', 'Nd']
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    ncols = max(round(len(params)/2), 1)
    nrows = ncols
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 10), dpi=500, sharex=True)
    
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = np.array([ax])
    
    y = cs['alpha_-1_neon_0.0'].ds['y'].values[1:-1]
    y_flipped = (y.max() - y)
    
    # Create inset axes
    inset_positions = ['lower right', 'lower left', 'upper right', 'upper left']
    inset_axes_list = []
    for i, axes in enumerate(ax):
        pos = inset_positions[i % len(inset_positions)]
        inset_ax = inset_axes(axes, width='45%', height='45%', loc=pos, borderpad=5)
        inset_ax.patch.set_facecolor('white')
        inset_ax.patch.set_alpha(0.9)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('gray')
        inset_axes_list.append(inset_ax)
    
    # Plot data for each alpha value
    for i in alpha_vals:
        linestyle, color, label = get_alpha_style(i)
        
        for j, value in enumerate(params):
            if value == 'q_cond':
                if i == 'snb':
                    profile = divq_integrate(cs[f'alpha_{i}_neon_{neon_value}'].ds.isel(t=-1), snb_int=True)
                else:
                    profile = replace_guards(spitzer_q(cs[f'alpha_{i}_neon_{neon_value}'].ds.isel(t=-1)))
                
                ax[j].plot(y_flipped[1:-1], profile[1:-1] * 1e-6, linestyle=linestyle, 
                          color=color, label=f'{label}', linewidth=2)
                inset_axes_list[j].plot(y_flipped[1:-1], profile[1:-1] * 1e-6, 
                                       linestyle=linestyle, color=color, linewidth=2)
                inset_axes_list[j].set_xscale(inset_xscale)
                
                ax[j].set_ylabel(r'$q_{\parallel, e}$ (MW/m$^2$)', fontsize=14)
                ax[j].set_ybound(0, 500)
                inset_axes_list[j].set_ybound(0, 500)
                inset_axes_list[j].set_ylabel('MW/m$^2$')
                
            elif value == 'Ne':
                profile = replace_guards(cs[f'alpha_{i}_neon_{neon_value}'].ds.isel(t=-1)[value].values)
                
                ax[j].plot(y_flipped, profile, linestyle=linestyle, color=color, 
                          label=f'{label}', linewidth=2)
                inset_axes_list[j].plot(y_flipped, profile, linestyle=linestyle, 
                                       color=color, linewidth=2)
                inset_axes_list[j].set_xscale(inset_xscale)
                
                ax[j].set_yscale('log')
                inset_axes_list[j].set_yscale('log')
                units = cs[f'alpha_{i}_neon_0.0'].ds.isel(t=-1)[value].units
                label_text = r'$n_e$'
                ax[j].set_ylabel(f'{label_text} (m$^{-3}$)', fontsize=14)
                ax[j].set_ybound(3e19, 1e20)
                inset_axes_list[j].set_ybound(3e19, 1e20)
                inset_axes_list[j].set_ylabel(r'm$^{-3}$')
                
            elif value == 'Nd':
                profile = replace_guards(cs[f'alpha_{i}_neon_{neon_value}'].ds.isel(t=-1)[value].values)
                
                ax[j].plot(y_flipped, profile, linestyle=linestyle, color=color, 
                          label=f'{label}', linewidth=2)
                inset_axes_list[j].plot(y_flipped, profile, linestyle=linestyle, 
                                       color=color, linewidth=2)
                inset_axes_list[j].set_xscale(inset_xscale)
                
                ax[j].set_yscale('log')
                inset_axes_list[j].set_yscale('log')
                units = cs[f'alpha_{i}_neon_0.0'].ds.isel(t=-1)[value].units
                label_text = r'$n_d$'
                ax[j].set_ylabel(f'{label_text} (m$^{-3}$)', fontsize=14)
                ax[j].set_ybound(1e12, 1e20)
                inset_axes_list[j].set_ybound(1e12, 1e20)
                inset_axes_list[j].set_ylabel(r'm$^{-3}$')
                
            elif value == 'Te':
                profile = replace_guards(cs[f'alpha_{i}_neon_{neon_value}'].ds.isel(t=-1)[value].values)
                
                ax[j].plot(y_flipped, profile, linestyle=linestyle, color=color, 
                          label=f'{label}', linewidth=2)
                inset_axes_list[j].plot(y_flipped, profile, linestyle=linestyle, 
                                       color=color, linewidth=2)
                inset_axes_list[j].set_xscale(inset_xscale)
                
                units = cs[f'alpha_{i}_neon_0.0'].ds.isel(t=-1)[value].units
                ax[j].set_ylabel(r'$T_e$ (eV)', fontsize=14)
                ax[j].set_ybound(75, 250)
                inset_axes_list[j].set_ybound(75, 150)
                inset_axes_list[j].set_ylabel(r'eV')
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    for axes in ax:
        axes.set_xscale('linear')
        axes.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)
        axes.tick_params(axis='both', which='minor', labelsize=10, width=1, length=4)
        for spine in axes.spines.values():
            spine.set_linewidth(1.2)
    
    ax[2].set_xlabel('Parallel distance to target (m)', fontsize=14)
    ax[3].set_xlabel('Parallel distance to target (m)', fontsize=14)
    
    for axes in ax:
        axes.set_xbound(-1, 70.1)
    
    # Set x-bounds for inset plots
    for inset_ax in inset_axes_list:
        if inset_xscale == 'log':
            x_min = max(0.1, np.min(y_flipped[y_flipped > 0]))
            x_max = np.max(y_flipped)
            inset_ax.set_xbound(x_min, x_max)
        else:
            inset_ax.set_xbound(-0.1, 2)
        
        inset_ax.tick_params(labelsize=10, width=1, length=4)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.2)
    
    # Create legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), 
              bbox_to_anchor=(0.5, 1.02), fontsize=14, frameon=True, 
              fancybox=True, shadow=True)
    fig.subplots_adjust(top=0.80, hspace=0.5, wspace=0.5)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_2_8_profiles_neon_{neon_value}.png', 
                dpi=500, bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_2_8_profiles_neon_{neon_value}.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2/8 to {output_dir}/figure_2_8_profiles_neon_{neon_value}.png")


def figure_3_target_temp_vs_neon(output_dir='figures'):
    """
    Generate Figure 3: Target Temperature vs Neon concentration
    """
    print("Generating Figure 3: Target Temperature vs Neon")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=500)
    
    neon_vals_array = np.array(neon_vals)
    
    for i in alpha_vals:
        target_temp_list = []
        upstream_temp_list = []
        neon_radiation_list = []
        linestyle, color, label = get_alpha_style(i)
        
        for j in neon_vals:
            print(f"Loading {i} {j}")
            target_temp = replace_guards(cs[f'alpha_{i}_neon_{j}'].ds.isel(t=-1)['Te'].values)[-1]
            target_temp_list.append(target_temp)
            upstream_temp = replace_guards(cs[f'alpha_{i}_neon_{j}'].ds.isel(t=-1)['Te'].values)[3]
            upstream_temp_list.append(upstream_temp)
            neon_radiation = -1 * sum(replace_guards(cs[f'alpha_{i}_neon_{j}'].ds.isel(t=-1)['Rneon'].values) * 
                                     replace_guards(cs[f'alpha_{i}_neon_{j}'].ds.isel(t=-1)['dv'].values))
            neon_radiation_list.append(neon_radiation)
        
        ax.plot(np.array(neon_vals_array)*100, target_temp_list, linestyle=linestyle, 
               color=color, label=f'{label}', markersize=10, marker='X')
    
    ax.axhline(5, color='black', linestyle='--', label='5 eV')
    ax.set_xlabel('Neon concentration (%)')
    ax.set_ylabel('Target temperature (eV)')
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_3_target_temp_vs_neon.png', dpi=500, bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_3_target_temp_vs_neon.svg', format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_dir}/figure_3_target_temp_vs_neon.png")


def figure_4_adas_curve(output_dir='figures'):
    """
    Generate Figure 4: ADAS neon radiation cooling curve
    """
    print("Generating Figure 4: ADAS neon radiation curve")
    
    Te = np.linspace(0, 500, 1000)
    result = neon_adas_curve(Te)
    
    max_index = np.argmax(result)
    print(f"Maximum at Te = {Te[max_index]} eV")
    
    fig, axis = plt.subplots(1, 1, figsize=(10, 6), dpi=500)
    
    axis.plot(Te, result, label='Neon ADAS Curve', linewidth=linewidth, color='blue')
    axis.set_xlabel('Te (eV)')
    axis.set_ylabel(r'Aggregate Neon Radiation (Wm$^{3}$)')
    axis.set_xbound(0, 500)
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_4_adas_curve.png', dpi=500, bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_4_adas_curve.svg', format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {output_dir}/figure_4_adas_curve.png")


def figure_5_profiles_colormap(neon_val=0.03, output_dir='figures'):
    """
    Generate Figure 5: Profiles color map
    """
    print(f"Generating Figure 5: Profiles colormap for neon={neon_val}")
    
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=500)
    
    # Create inset axes
    ax_inset = inset_axes(ax, width="80%", height="80%", 
                         bbox_to_anchor=(0.45, 0.05, 0.5, 0.5),
                         bbox_transform=ax.transAxes)
    
    S_II = np.linspace(0, 72, 500)
    T_e = np.linspace(0, 250, 500)
    S_II_grid, T_e_grid = np.meshgrid(S_II, T_e)
    Z = neon_adas_curve(T_e_grid)
    cmap = 'Purples'
    norm = Normalize(vmin=np.min(Z), vmax=np.max(Z))
    ax.pcolormesh(S_II_grid, T_e_grid, Z, shading='auto', norm=norm, cmap=cmap)
    ax_inset.pcolormesh(S_II_grid, T_e_grid, Z, shading='auto', norm=norm, cmap=cmap)
    fig.colorbar(ax.collections[0], label=r'$L_z$(Wm$^{3}$)')
    
    # Zoomed region
    S_II = np.linspace(64, 72, 500)
    T_e = np.linspace(0, 250, 500)
    S_II_grid, T_e_grid = np.meshgrid(S_II, T_e)
    Z = neon_adas_curve(T_e_grid)
    norm = Normalize(vmin=np.min(Z), vmax=np.max(Z))
    
    y = cs[f'alpha_-1_neon_{neon_val}'].ds['y'].values[1:-1]
    y_flipped = (np.max(y) - y)
    
    print(f'neon = {neon_val}')
    
    for i in alpha_vals:
        linestyle, color, label = get_alpha_style(i)
        profile = replace_guards(cs[f'alpha_{i}_neon_{neon_val}'].ds.isel(t=-1)['Te'].values)
        ax.plot(y_flipped, profile, linestyle=linestyle, color=color, 
               label=f'{label}', linewidth=linewidth)
        ax_inset.plot(y_flipped, profile, linestyle=linestyle, color=color, linewidth=linewidth)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
    ax.set_ylabel(r'Te (eV)')
    ax.set_xlabel(r'Parallel distance to target (m)')
    
    ax_inset.set_xbound(0, 2)
    ax_inset.set_ybound(50, 125)
    fig.suptitle(f'{neon_val}% Neon')
    
    ax_inset.tick_params(axis='both', colors='white')
    for spine in ax_inset.spines.values():
        spine.set_color('white')
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_5_profiles_colormap_neon_{neon_val}.png', 
                dpi=500, bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_5_profiles_colormap_neon_{neon_val}.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {output_dir}/figure_5_profiles_colormap_neon_{neon_val}.png")


def figure_6_neon_radiation_profile(neon=0.03, output_dir='figures'):
    """
    Generate Figure 6: Neon radiation profile for specified Neon concentration
    """
    print(f"Generating Figure 6: Neon radiation profile for neon={neon}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    y = cs['alpha_-1_neon_0.0'].ds['y'].values[1:-1]
    
    for i in alpha_vals:
        linestyle, color, label = get_alpha_style(i)
        param = (replace_guards(cs[f'alpha_{i}_neon_{neon}'].ds.isel(t=-1)['Rneon'].values) * 
                replace_guards(cs[f'alpha_{i}_neon_{neon}'].ds.isel(t=-1)['dv'].values))
        yflipped = (np.max(y) - y)
        ax.plot((yflipped)[1:], (param*-1e-6)[1:], linestyle=linestyle, 
               color=color, label=f'{label}')
    
    ax.legend()
    ax.set_xlabel('Parallel distance to target (m)')
    ax.set_ylabel('Neon radiation (MW)')
    ax.set_xbound(-0.5, 70)
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_6_neon_radiation_neon_{neon}.png', 
                dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_6_neon_radiation_neon_{neon}.svg', 
                format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 6 to {output_dir}/figure_6_neon_radiation_neon_{neon}.png")


def figure_7_collisionality(output_dir='figures'):
    """
    Generate Figure 7: Collisionality calculation
    """
    print("Generating Figure 7: Collisionality")
    
    plt.rcParams.update({'font.size': 10})
    linewidth = 3
    markersize = 10
    
    plt.style.use('default')
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.grid'] = True
    plt.rcParams.update({'font.size': 16})
    
    neon_vals_selected = [0.03, 0.04]
    models = ['-1', 'snb']
    model_names = ['SH', 'SNB']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), dpi=500)
    
    # Share y-axes within rows
    for col in range(1, 4):
        axes[0, col].sharey(axes[0, 0])
    for col in range(1, 4):
        axes[1, col].sharey(axes[1, 0])
    
    # Create twin axes for density
    density_axes = []
    for col in range(4):
        ax_ne = axes[1, col].twinx()
        density_axes.append(ax_ne)
        if col > 0:
            ax_ne.sharey(density_axes[0])
    
    col_idx = 0
    for neon_idx, neon_conc in enumerate(neon_vals_selected):
        for model_idx, model in enumerate(models):
            j = model
            
            if j == '-1':
                model_title = 'SH'
            elif j == 'snb':
                model_title = 'SNB'
            else:
                model_title = f'α = {j}'
            
            Te = replace_guards(cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['Te'])
            y = cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['pos'].values[2:-2]
            Ne = replace_guards(cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['Ne'])
            
            # Calculate non-locality metrics
            lambda_ee = ((Te**2) * 10**(16) / Ne) / np.sqrt(2*np.pi)
            Ls = np.max(cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['pos'].values[2:-2])
            
            y_flipped = (cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['pos'].values[1:-1].max() - 
                        cs[f'alpha_{j}_neon_{neon_conc}'].ds.isel(t=-1)['pos'].values[1:-1])
            
            print(f"Model: {model_title}, Neon: {neon_conc*100}% - Lengths: y_flipped={len(y_flipped)}, Te={len(Te)}, Ne={len(Ne)}")
            
            grad_T = np.gradient(Te, y_flipped)
            Lt = np.abs(Te/grad_T)
            
            lt_over_lambda = (Lt/lambda_ee)
            ls_over_lambda = (Ls/lambda_ee)
            
            # Using constant upstream temperature
            lambda_ee_const = ((Te[2]**2) * 10**(16) / Ne) / np.sqrt(2*np.pi)
            lt_over_lambda_const = (Lt/lambda_ee_const)
            ls_over_lambda_const = (Ls/lambda_ee_const)
            
            ax_top = axes[0, col_idx]
            ax_temp = axes[1, col_idx]
            ax_ne = density_axes[col_idx]
            
            # Top row - Non-locality metrics
            ax_top.plot(y_flipped, lt_over_lambda, '-', color='blue', label=r'$K_n^{-1}$')
            ax_top.plot(y_flipped, ls_over_lambda, '-', color='black', label=r'$\nu_{sol}^*$')
            ax_top.plot(y_flipped, ls_over_lambda_const, '--', color='black', label=r'$\nu_{sol, Tu}^*$')
            ax_top.axhline(100, color='red', linestyle='--')
            
            ax_top.set_ylim(0, 200)
            ax_top.set_title(f'{model_title} {neon_conc*100}% neon')
            ax_top.set_xticklabels([])
            ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            if col_idx == 0:
                ax_top.set_ylabel('Non-locality metric')
                ax_top.legend(loc='upper right')
            else:
                ax_top.tick_params(axis='y', left=False, labelleft=False)
            
            # Bottom row - Temperature and density profiles
            ax_temp.plot(y_flipped, Te, '-', color='red', label='Te (eV)')
            ax_ne.plot(y_flipped, Ne, '-', color='blue', label=r'$n_e$ (m$^{-3}$)')
            
            ax_ne.set_yscale('log')
            ax_ne.set_ybound(1e19, 1e21)
            ax_ne.grid(False)
            
            if col_idx == 0:
                ax_temp.set_ylabel('Te (eV)', color='red')
                ax_temp.tick_params(axis='y', labelcolor='red')
                ax_ne.tick_params(axis='y', right=False, labelright=False)
            elif col_idx == 3:
                ax_ne.set_ylabel(r'$n_e$ (m$^{-3}$)', color='blue')
                ax_ne.tick_params(axis='y', labelcolor='blue')
                ax_temp.tick_params(axis='y', left=False, labelleft=False)
            else:
                ax_temp.tick_params(axis='y', left=False, labelleft=False)
                ax_ne.tick_params(axis='y', right=False, labelright=False)
            
            ax_temp.set_xlabel('Parallel distance to target (m)')
            
            if col_idx == 0:
                lines1, labels1 = ax_temp.get_legend_handles_labels()
                lines2, labels2 = ax_ne.get_legend_handles_labels()
                ax_temp.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            col_idx += 1
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/figure_7_collisionality.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{output_dir}/figure_7_collisionality.png', format='png', dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7 to {output_dir}/figure_7_collisionality.png")


def main():
    """Main function to generate all figures"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate figures for flux limiter detachment paper')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures (default: figures)')
    parser.add_argument('--skip-load', action='store_true',
                       help='Skip data loading (use if data already loaded)')
    parser.add_argument('--figures', nargs='+', type=int, default=[2, 3, 4, 5, 6, 7],
                       help='List of figure numbers to generate (default: all)')
    parser.add_argument('--neon-value', type=str, default='0.0',
                       help='Neon value for figures 2/8 and 5 (default: 0.0)')
    
    args = parser.parse_args()
    
    # Load data if not skipping
    if not args.skip_load:
        load_data()
    
    # Generate requested figures
    if 2 in args.figures or 8 in args.figures:
        figure_2_8_profiles(neon_value=args.neon_value, output_dir=args.output_dir)
    
    if 3 in args.figures:
        figure_3_target_temp_vs_neon(output_dir=args.output_dir)
    
    if 4 in args.figures:
        figure_4_adas_curve(output_dir=args.output_dir)
    
    if 5 in args.figures:
        figure_5_profiles_colormap(neon_val=float(args.neon_value), output_dir=args.output_dir)
    
    if 6 in args.figures:
        figure_6_neon_radiation_profile(neon=float(args.neon_value), output_dir=args.output_dir)
    
    if 7 in args.figures:
        figure_7_collisionality(output_dir=args.output_dir)
    
    print("\nAll requested figures generated successfully!")


if __name__ == '__main__':
    main()

