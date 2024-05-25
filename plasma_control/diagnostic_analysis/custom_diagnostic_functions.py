import numpy as np
import matplotlib.pyplot as plt
from mast.mast_client import ListType
import pyuda
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys, os
from scipy.signal import medfilt
import warnings
from matplotlib.animation import FuncAnimation

# Grab P Ryans LP functions
lp_module_path = '/home/cl6305/Documents/data_access/mastu_exhaust_analysis/mastu_exhaust_analysis'
if lp_module_path not in sys.path:
    sys.path.append(lp_module_path)

from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, compare_shots, probe_array
# J Lovell bolo funcs
from mastu_exhaust_analysis.pyBolometer import Bolometer

""" LP diagnostics """

def lp_asymmetry(shot_number, sector, parameter, smoothing=False, 
                        kernel_size = 21, output_time = False):
    """
    Calculates the asymmetry between the upper and lower Langmuir probe measurements.

    Args:
        shot_number (int): The shot number to retrieve data for
        sector (int): Langmuir probe sector
        parameter (str): The measurement parameter to retrieve (e.g., 'isat', 'jsat', 'Te')
        smoothing (bool): Apply median smoothing to the data
        kernal_size (int): kernal size for smoothing

    Returns:
        np.array: {Asymmetry: Asymmetry values computed as (upper - lower) / (upper + lower),
        time: time array}

    """
    filename = f'/common/uda-scratch/pryan/elp0{shot_number}.nc'
    lp_data = LangmuirProbe(filename=filename)
    
    # Retrieving data for upper and lower probes
    upper_data_group = getattr(lp_data, f's{sector}_upper_data')
    lower_data_group = getattr(lp_data, f's{sector}_lower_data')

    # Retrieving time (should be same for upper and lower)
    probe_time = getattr(upper_data_group, 'time')

    # Fetching specific parameters
    probe_data_upper = getattr(upper_data_group, parameter)
    probe_data_lower = getattr(lower_data_group, parameter)

    # Compute maximum along the time axis
    max_probe_upper = np.nanmean(probe_data_upper, axis=1)
    max_probe_lower = np.nanmean(probe_data_lower, axis=1)


    # Calculating asymmetry
    up_down_asymmetry = (max_probe_upper - max_probe_lower) / (max_probe_upper + max_probe_lower)

    # Optional smoothing
    if smoothing:
        # kernel_size = 21  # Kernel size for median filter; adjust as needed
        up_down_asymmetry = medfilt(up_down_asymmetry, kernel_size=kernel_size)

    if output_time:
        return up_down_asymmetry, probe_time

    else:
        return up_down_asymmetry

""" D-alpha Diagnostics"""
def d_alpha_divertor_asymmetry(shot_number, site_line, output_time = False):
    client=pyuda.Client()

    if site_line == 'OSP':
        upper_signal = client.get(f'/XIM/DA/HU10/OSP', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HL02/OSP', shot_number).data
        time = client.get(f'/XIM/DA/HU10/OSP', shot_number).time.data
    elif site_line == 'SXD':
        upper_signal = client.get(f'/XIM/DA/HU10/SXD', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HL02/SXD', shot_number).data
        time = client.get(f'/XIM/DA/HU10/SXD', shot_number).time.data
    elif site_line == 'ISP':
        # upper_signal = client.get(f'/XIM/DA/HU10/ISP', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HE05/ISP/L', shot_number).data
        time = client.get(f'/XIM/DA/HE05/ISP/L', shot_number).time.data
    else:
        print('Invalid site line, valid lines of site are: OSP, SXD or ISP')
        return

    asymmetry = (upper_signal - lower_signal) / (upper_signal + lower_signal)

    if output_time:
        return asymmetry, time
    
    else:
        return asymmetry


def d_alpha_signal(shot_number, site_line, output_time = False):
    client=pyuda.Client()

    if site_line == 'OSP':
        upper_signal = client.get(f'/XIM/DA/HU10/OSP', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HL02/OSP', shot_number).data
        time = client.get(f'/XIM/DA/HU10/OSP', shot_number).time.data
    elif site_line == 'SXD':
        upper_signal = client.get(f'/XIM/DA/HU10/SXD', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HL02/SXD', shot_number).data
        time = client.get(f'/XIM/DA/HU10/SXD', shot_number).time.data
    elif site_line == 'ISP':
        upper_signal = client.get(f'/XIM/DA/HU10/ISP', shot_number).data
        lower_signal = client.get(f'/XIM/DA/HL02/ISP', shot_number).data
        time = client.get(f'/XIM/DA/HU10/ISP', shot_number).time.data
    else:
        print('Invalid site line, valid lines of site are: OSP, SXD or ISP')
        return

    if output_time:
        return upper_signal, lower_signal, time
    
    else:
        return upper_signal, lower_signal


""" Magnetic Diagnostics """
def magnetic_axis_efit(shot_number, output_time = False, normalise = False, coordinate = 'Z'):

    client=pyuda.Client()
    mag_time = client.get(f'/epm/output/globalParameters/magneticAxis/{coordinate}',shot_number).time.data
    mag_axis = client.get(f'/epm/output/globalParameters/magneticAxis/{coordinate}',shot_number).data

    if normalise:
        abs_max = np.max(np.abs(mag_axis))
        mag_axis = mag_axis / abs_max

    if output_time:
        return mag_axis,mag_time
    
    else:
        return mag_axis


def magnetic_axis_zc(shot_number, output_time = False, normalise = False, trim = True):

    client=pyuda.Client()
    np.seterr(divide='ignore', invalid='ignore')

    time = client.get(f'/xzc/zcon/zip',shot_number).time.data
    zip = client.get(f'/xzc/zcon/zip',shot_number).data 
    ip = client.get(f'/xzc/zcon/ip',shot_number).data

    print(ip)
    print(len(ip))

    # Have to divide by ip to get z
    z = zip/ip

    z = np.divide(zip,ip, out=np.zeros_like(zip), where=ip!=0)

    # We know the pulse starts at 0, so we can set all values before this to 0
    z[time < 0.01] = 0

    derivative = np.diff(z)
    unphysical_index = np.where(np.abs(derivative) > np.mean(np.abs(derivative)) + 3 * np.std(np.abs(derivative)))[0]
    if len(unphysical_index) > 0:
        first_unphysical_index = unphysical_index[0] + 1  # +1 to account for diff reducing length by 1
        z[first_unphysical_index -1 :] = 0

    if normalise:
        abs_max = np.max(np.abs(zip))
        zip = zip / abs_max

    if trim:
        # Identify indices where values are unphysical
        outlier_indices = np.where((z > 0.15) | (z < -0.15))[0]

        # Generate indices for valid data points
        valid_indices = np.delete(np.arange(len(z)), outlier_indices)

        # Interpolate to replace outliers
        interpolator = interp1d(valid_indices, z[valid_indices], kind='linear', fill_value="extrapolate")
        z[outlier_indices] = interpolator(outlier_indices)

    if output_time:
        return z, time
    
    else:
        return z

def plot_equib(shot_number, ):
    client=pyuda.Client()
    psiNorm_data = client.get('EPM/OUTPUT/PROFILES2D/poloidalFlux', shot_number).data
    time_data = client.get('EPM/OUTPUT/PROFILES2D/poloidalFlux', shot_number).time.data

    limR = client.geometry('/limiter/efit', 45470).data.R
    limZ = client.geometry('/limiter/efit', 45470).data.Z

    print(np.shape(psiNorm_data))

    x= np.linspace(0.06, 2, 65)

    y = np.linspace(-2.2, 2.2, 65)

    X, Y = np.meshgrid(y,x)

    fig,ax = plt.subplots()
    ax.contour(Y,X, psiNorm_data[100], levels = 1, colors = 'red')
    ax.contour(Y,X, psiNorm_data[100], level = 100, cmap = 'plasma', alpha = 0.5)
    ax.plot(limR, limZ, color = 'black')
    
    ax.set_aspect('equal')
    plt.show()
    
def plot_equib_animate(shot_number):
    client = pyuda.Client()
    psiNorm_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).data
    time_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).time.data

    limR = client.geometry('/limiter/efit', shot_number).data.R
    limZ = client.geometry('/limiter/efit', shot_number).data.Z

    print(np.shape(psiNorm_data))

    x = np.linspace(0.06, 2, 65)
    y = np.linspace(-2.2, 2.2, 65)
    X, Y = np.meshgrid(y, x)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Initial plot setup with the first frame of data
    contour_filled = ax.contourf(Y, X, psiNorm_data[0], levels=100, cmap='plasma', alpha=0.5)
    contour_line = ax.contour(Y, X, psiNorm_data[0],1, levels=5, cmap = 'plasma', alpha=1)
    limiter_line, = ax.plot(limR, limZ, color='black')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)  # Prepare text location

    def update(frame):
        nonlocal contour_filled, contour_line  # Declare these as nonlocal to modify existing variables
        # Clear previous contours
        for coll in contour_filled.collections:
            coll.remove()
        for coll in contour_line.collections:
            coll.remove()

        # Redraw with new time slice data
        contour_filled = ax.contourf(Y, X, psiNorm_data[frame], levels=100, cmap='plasma', alpha=0.5)
        contour_line = ax.contour(Y, X, psiNorm_data[frame], levels=[1], colors='red', alpha = 1)

        # Update the time display
        time_text.set_text(f'Time: {time_data[frame]:.2f} s')  # Display current time, formatted to 2 decimal places

        return contour_filled.collections + contour_line.collections + [time_text]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(time_data), interval=50, blit=True)

    plt.show()
    return anim



""" Bolometry Diagnostics """

def bolo_sxd_asymmetry(shot_number, output_time = False):
    bolo = Bolometer(shot_number)


    upper_sxd_bolo = bolo.sxd_upper_prad

    lower_sxd_bolo = bolo.sxd_lower_prad
    # print(lower_sxd_bolo)

    # max_upper = np.nanmax(upper_sxd_bolo)
    # max_lower = np.nanmax(lower_sxd_bolo)

    bolo_time = bolo.time

    asymmetry = (upper_sxd_bolo - lower_sxd_bolo)/(upper_sxd_bolo + lower_sxd_bolo)

    if output_time:
        return asymmetry, bolo_time

    else:
        return asymmetry

def core_bolo_xpoint_asymmetry(shot_number, output_time = False):
    # Bolo data
    client=pyuda.Client()
    time_arr = client.get('ABM/SXDL/PRAD', shot_number).time.data
    core_brightness = client.get('ABM/CORE/BRIGHTNESS', shot_number).data

    channel3 = core_brightness[:, 2]
    channel4 = core_brightness[:, 3]

    channel13 = core_brightness[:, 13]
    channel14 = core_brightness[:, 14] 

    upper_mean_brightness = (channel3 + channel4)/2
    lower_mean_brightness = (channel13 + channel14)/2

    asymmetry_arr = (upper_mean_brightness - lower_mean_brightness) / (upper_mean_brightness + lower_mean_brightness)

    if output_time:
        return asymmetry_arr, time_arr

    else: 
        return asymmetry_arr



""" Combined diagnostics """



def plot_equib_with_traces_defunct(shot_number, psi_value = 1):
    client = pyuda.Client()
    psiNorm_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).data
    time_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).time.data

    # Fetch diagnostic data along with their time data
    sxd_bolo_asymmetry, sxd_time = bolo_sxd_asymmetry(shot_number, output_time=True)
    lang_asymmetry, lang_time = lp_asymmetry(shot_number, 4, 'isat', smoothing=True, output_time=True)
    mag_axis, mag_time = magnetic_axis_zc(shot_number, output_time=True, normalise=False)
    core_bolo_asymmetry, core_bolo_time = core_bolo_xpoint_asymmetry(shot_number, output_time=True)

    x = np.linspace(0.06, 2, 65)
    y = np.linspace(-2.2, 2.2, 65)
    X, Y = np.meshgrid(y, x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.set_aspect('equal')

    # Initial plot for magnetic axis and contour on ax1
    contour_line = ax1.contour(Y, X, psiNorm_data[0], levels=[psi_value], colors='red')

    # Initial plot for diagnostic data on ax2
    ax2.plot(sxd_time, sxd_bolo_asymmetry, label='SXDB Asymmetry')
    ax2.plot(lang_time, lang_asymmetry, label='Langmuir Probe Asymmetry')
    ax2.plot(core_bolo_time, core_bolo_asymmetry, label='Core Bolometry Asymmetry')
    ax3 = plt.twinx(ax2)
    ax3.plot(mag_time, mag_axis, color = 'red', linestyle = '--', label = 'magnetic_axis')
    
    ax2.set_xbound(0,0.8)
    ax2.set_ybound(-1.2,1.2)

    ax3.set_ylim(-0.12,0.12)

    ax2.legend()

    # Vertical line on ax2 for the current time
    vline = ax2.axvline(x=time_data[0], color='r', linestyle='--')

    def update(frame):
        ax1.clear()
        ax1.set_aspect('equal')
        
        # Generate contour plot only if levels are within the data range
        if np.nanmin(psiNorm_data[frame]) <= psi_value <= np.nanmax(psiNorm_data[frame]):
            contour_line = ax1.contour(Y, X, psiNorm_data[frame], levels=[psi_value], colors='red')
            ax1.clabel(contour_line, inline=True, fontsize=10)
        else:
            print(f"No valid contour levels for frame {frame} with psi_value {psi_value}")

        # Update vertical line on diagnostic plot
        vline.set_xdata([time_data[frame], time_data[frame]])

        return ax1,vline


    anim = FuncAnimation(fig, update, frames=len(time_data), interval=50, blit=False)

    plt.show()
    return anim


def plot_equib_with_traces(shot_number, psi_value=1, plot_lang_asymmetry=True, save_gif=False, sector = 4):
    client = pyuda.Client()
    psiNorm_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).data
    time_data = client.get('EPM/OUTPUT/PROFILES2D/PSINORM', shot_number).time.data

    # Fetch diagnostic data along with their time data
    sxd_bolo_asymmetry, sxd_time = bolo_sxd_asymmetry(shot_number, output_time=True)
    if plot_lang_asymmetry:
        lang_asymmetry, lang_time = lp_asymmetry(shot_number, sector, 'isat', smoothing=True, output_time=True)
    mag_axis, mag_time = magnetic_axis_zc(shot_number, output_time=True, normalise=False)
    core_bolo_asymmetry, core_bolo_time = core_bolo_xpoint_asymmetry(shot_number, output_time=True)

    limR = client.geometry('/limiter/efit', shot_number).data.R
    limZ = client.geometry('/limiter/efit', shot_number).data.Z

    x = np.linspace(0.06, 2, 65)
    y = np.linspace(-2.2, 2.2, 65)
    X, Y = np.meshgrid(y, x)

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(20, 12))
    ax1.set_aspect('equal')

    fig.suptitle(f'Diagnostic asymmetry vs magnetic axis Z position \n shot {shot_number}')
    ax1.set_xlabel('R (M)')
    ax1.set_ylabel('Z (M)')

    # Initial plot setup for the contour and magnetic axis on ax1 (right plot)
    contour_filled = ax1.contourf(Y, X, psiNorm_data[0], levels=100, cmap='plasma', alpha=0.5)
    contour_line = ax1.contour(Y, X, psiNorm_data[0], levels=[psi_value], colors='red')

    # Initial plot for diagnostic data on ax2 (left plot)
    ax2.plot(sxd_time, sxd_bolo_asymmetry, label='SXDB Asymmetry')
    if plot_lang_asymmetry:
        ax2.plot(lang_time, lang_asymmetry, label='Langmuir Probe Asymmetry')
    ax2.plot(core_bolo_time, core_bolo_asymmetry, label='Core Bolometry Asymmetry')
    ax3 = ax2.twinx()
    ax3.plot(mag_time, mag_axis, color='red', linestyle='--', label='Magnetic Axis')

    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Asymmetry')
    ax3.set_ylabel('Magnetic Axis Z Position (m)')
    ax2.set_xbound(0, np.max(mag_time))
    ax2.set_ybound(-1.2, 1.2)
    ax3.set_ylim(-0.12, 0.12)

    # Vertical line on ax2 for the current time
    vline = ax2.axvline(x=time_data[0], color='r', linestyle='--')

    def update(frame):
        ax1.clear()
        ax1.set_aspect('equal')
        contour_filled = ax1.contourf(Y, X, psiNorm_data[frame], levels=100, cmap='plasma', alpha=0.5)
        contour_line = ax1.contour(Y, X, psiNorm_data[frame], levels=[psi_value], colors='red')
        ax1.plot(limR, limZ, color='black')
        time_text = ax1.text(0.05, 0.95, f'Time: {time_data[frame]:.2f}s', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
        vline.set_xdata([time_data[frame], time_data[frame]])
        return [contour_filled.collections + contour_line.collections + [time_text, vline]]

    anim = FuncAnimation(fig, update, frames=len(time_data), interval=50, blit=False)

    if save_gif:
        anim.save(f'shot_{shot_number}_with_data_traces_animation.gif', writer='imagemagick', fps=10)

    plt.show()
    return anim




if __name__=='__main__':
    shot=49260

    # bolo, time = bolo_sxd_asymmetry(shot, output_time=True)

    # print(bolo, time)

    # plt.plot(time,bolo)

    # core, time = core_bolo_xpoint_asymmetry(shot, output_time=True)

    # print(core,time)

    # plot_equib_animate(shot)

    # plot_equib_with_traces(shot, save_gif= False, sector = 4)

    # d_alpha_asymmetry, time = d_alpha_divertor_asymmetry(shot, 'SXD', output_time = True)

    mag_axis, time = magnetic_axis_zc(shot, output_time = True)

    # print(d_alpha_asymmetry, time)
