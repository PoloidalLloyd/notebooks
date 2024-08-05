import numpy as np
import matplotlib.pyplot as plt
from mast.mast_client import ListType
import pyuda
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys, os
from scipy.optimize import curve_fit

lp_module_path = '/home/cl6305/Documents/data_access/mastu_exhaust_analysis/mastu_exhaust_analysis'
if lp_module_path not in sys.path:
    sys.path.append(lp_module_path)

from mastu_exhaust_analysis.pyLangmuirProbe import LangmuirProbe, compare_shots, probe_array

notebook_functions_path = '/home/cl6305/Documents/data_access/notebooks/plasma_control/diagnostic_analysis/'
if notebook_functions_path not in sys.path:
    sys.path.append(notebook_functions_path)

from custom_diagnostic_functions import *
import pyuda
client=pyuda.Client()
import matplotlib.pyplot as plt
import numpy as np

class DZFit:
    def __init__(self, data_set_x, data_set_y):
        self.data_set_x = data_set_x
        self.data_set_y = data_set_y
        self.params = None
        self.pcov = None
        self.fitted_data = None

    def exp_function(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def fit(self, output_error=False, output_params=False):
        popt, pcov = curve_fit(self.exp_function, self.data_set_x, self.data_set_y, p0=(1, 0.1, 1))
        self.params = popt
        self.pcov = pcov

        # Generate fitted data using the optimal parameters
        self.fitted_data = self.exp_function(self.data_set_x, *popt)

        if output_error:
            # standard deviations
            perr = np.sqrt(np.diag(pcov))
            # Calculate partial derivatives
            dfdx_a = np.exp(popt[1] * self.data_set_x)
            dfdx_b = popt[0] * self.data_set_x * np.exp(popt[1] * self.data_set_x)
            dfdx_c = np.ones_like(self.data_set_x)
            
            # Calculate the confidence interval
            delta = np.sqrt((dfdx_a * perr[0])**2 + (dfdx_b * perr[1])**2 + (dfdx_c * perr[2])**2)
            
            if output_params:
                print(f'Fitted curve equation (params= a, b, c): y = {popt[0]:.2f}*exp({popt[1]:.2f}*x) + {popt[2]:.2f}')
                return self.fitted_data, self.params, delta
            else:
                return self.fitted_data, delta

        if output_params:
            return self.fitted_data, self.params
        else:
            return self.fitted_data




def exp_function(x, a, b, c):
    return a * np.exp(b * x) + c

def exp_fit(data_set_x, data_set_y, output_error=False, output_params = False):

    # def exp_function(x, a, b, c):
    #     return a * np.exp(b * x) + c

    popt, pcov = curve_fit(exp_function, data_set_x, data_set_y, p0=(1, 0.1, 1))

    # Optimal params
    a, b, c = popt

    # Generate fitted data using the optimal parameters
    fitted_data = exp_function(data_set_x, *popt)
    params_dict = {'a':a, 'b':b, 'c':c}
    if output_error == True:
        # standard deviations
        perr = np.sqrt(np.diag(pcov))
        # Calculate partial derivatives
        dfdx_a = np.exp(b * data_set_x)
        dfdx_b = a * data_set_x * np.exp(b * data_set_x)
        dfdx_c = 1
    
    # Calculate the confidence interval
        delta = np.sqrt((dfdx_a * perr[0])**2 + (dfdx_b * perr[1])**2 + (dfdx_c * perr[2])**2)
        if output_params == True:
            print(f'Fitted curve equation (params= a, b, c): y = {a:.2f}*exp({b:.2f}*x) + {c:.2f}')
            return fitted_data, params_dict, delta
        else:
            return fitted_data, delta

    if output_params == True:
        return fitted_data, params_dict
    else:
        return fitted_data

def align_and_interpolate(time_array1, data_set1, time_array2, data_set2, time_range=None):
    """
    Aligns and interpolates two datasets based on their corresponding time arrays,
    with an optional user-specified time range for trimming.

    Parameters:
    - time_array1: np.array, Time points for the first dataset.
    - data_set1: np.array, Data points corresponding to time_array1.
    - time_array2: np.array, Time points for the second dataset.
    - data_set2: np.array, Data points corresponding to time_array2.
    - time_range: tuple, optional, (start_time, end_time) for trimming the data. 
                  Defaults to the common range of the two arrays if not provided.

    Returns:
    - common_time_array: np.array, Aligned and interpolated common time array.
    - interpolated_data1: np.array, Interpolated data points for the first dataset.
    - interpolated_data2: np.array, Interpolated data points for the second dataset.
    """
    # 1. Determine the default common start and end times
    default_start_time = max(time_array1[0], time_array2[0])
    default_end_time = min(time_array1[-1], time_array2[-1])

    # 2. Use the provided time range if specified, otherwise use default
    if time_range is not None:
        start_time = max(default_start_time, time_range[0])
        end_time = min(default_end_time, time_range[1])
    else:
        start_time = default_start_time
        end_time = default_end_time

    # 3. Trim the time arrays and corresponding data sets
    trimmed_time1 = time_array1[(time_array1 >= start_time) & (time_array1 <= end_time)]
    trimmed_data1 = data_set1[(time_array1 >= start_time) & (time_array1 <= end_time)]

    trimmed_time2 = time_array2[(time_array2 >= start_time) & (time_array2 <= end_time)]
    trimmed_data2 = data_set2[(time_array2 >= start_time) & (time_array2 <= end_time)]

    # 4. Create a common time array for interpolation
    desired_length = min(len(trimmed_time1), len(trimmed_time2))
    common_time_array = np.linspace(start_time, end_time, desired_length)

    # 5. Interpolate data sets to the common time array
    interpolated_data1 = np.interp(common_time_array, trimmed_time1, trimmed_data1)
    interpolated_data2 = np.interp(common_time_array, trimmed_time2, trimmed_data2)

    return common_time_array, interpolated_data1, interpolated_data2

if __name__=='__main__':
   shot=49059  