import time
import sys
from EEDFB_Finite_Mode_Solver import EEDFB_Solver
from EEDFB_Finite_Mode_Solver import load_parameters
from EEDFB_Finite_Mode_Solver import save_parameters
import json
import pandas as pd
import os
import numpy as np
### This is explicitly for varying a non-trivial parameters (so not duty cycle, cladding thickness, or grating height)
### This only sweeps over the value in the params file 'results_indices'
### To vary DC, CT, GH, use results_Sweep.py

def Parameter_Sweep(filename_original):

    param_label = 'cleave_phase_shift'
    param_values = np.arange(0, 2* np.pi, 2*np.pi * 0.05)

    headers = [ param_label, 'rR', 'duty_cycle', 'cladding_thickness', 
            'grating_height', 'PmaxWPE_est_CW', 'kappa_DFB_L', 'S0gacterm', 
            'AReff', 'Homega_4', 'alpha_m', 'Jth', 'del_gth', 'eta_s_est', 
            'tau_photon', 'tau_stim_4', 'Guided_plot_abs'
    ]


    excel_out, _ = os.path.splitext(filename_original)
    excel_out += "_" + param_label + "_sweep.xlsx" 

    data = {header: [] for header in headers}

    params_sweep = True

    for val in param_values:
        print(f"Solving for {param_label} = {val}") 
        append_ = param_label + str(val)
            
        ### Overwriting the param value
        params = load_parameters("Data/" + filename_original)
        params[param_label] = val
        save_parameters(params, "Data/" + filename_original)
        
        
        results = EEDFB_Solver("Data/" + filename_original, params_sweep)
        data[param_label].append(val)

        #print(len(results))
        for header, result in zip(headers[1:], results):
            data[header].append(result)

    df = pd.DataFrame(data)
    df.to_excel(excel_out, index=False)


if __name__ == "__main__":
    filename = sys.argv[1]
    Parameter_Sweep(filename)
