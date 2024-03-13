import sys
from DFB_Finite_Mode_Solver import DFB_Solver
from DFB_Finite_Mode_Solver import load_parameters
from DFB_Finite_Mode_Solver import save_parameters
from Post_Process_EE import default_Calculations, find_results_Args
from Post_Process_EE import index_Plot
import os
import pandas as pd
import time
import numpy as np
### This is explicitly for varying a trivial parameters (so duty cycle, cladding thickness, or grating height)
### Best method is to run EEDFB_Finite_Mode_Solver.py first, find the indices to target, and then run this to grab the csv file export.
def Results_Sweep(filename_original):

    excel_out, _ = os.path.splitext(filename_original)
    excel_out ="Data/" + excel_out + ".xlsx"

    headers = ['rR', 'duty_cycle', 'cladding_thickness', 
            'grating_height', 'PmaxWPE_est_CW', 'kappa_DFB_L', 'S0gacterm', 
            'AReff', 'Homega_4', 'alpha_m', 'alpha_opt', 'Jth', 'del_gth', 'eta_s_est', 
            'tau_photon', 'tau_stim_4', 'Field_end'
    ]
    data = {header: [] for header in headers}
    params = load_parameters("Data/" + filename_original)

    ### so that the end term is included in the range
    eps = 1e-05
    ### If you want to use a single value, just set the end-bound to the start-bound+eps and it should only be one value
    sweep_values = [
        np.arange(2.9, 3.1 + eps, 0.1),
        np.arange(0.25, 0.27 + eps, 0.01),
        np.arange(0.5, 0.5 + eps, 0.05)    
    ]


    if params['plot_Indices']:
        index_Plot(params['cladding_thickness'], params['grating_height'])
        
    for i_ in sweep_values[0]:
        for j_ in sweep_values[1]:
            for k_ in sweep_values[2]:
                
                #start = time.time()
                #print(i)

                params['results_values'] = [i_, j_, k_]
                
                if params['modes_solved']:
                    results = default_Calculations(params, False)
                else:
                    save_parameters(params, "Data/" + filename_original)
                    results = DFB_Solver("Data/" + filename_original)
                    params = load_parameters("Data/" + filename_original)
                                    
                for header, result in zip(headers, results):
                    data[header].append(result)
                #print(f"Mode solver time elapsed is {time.time() - start} seconds.") 


    df = pd.DataFrame(data)
    df.to_excel(excel_out, index=False)

if __name__ == "__main__":
    filename = sys.argv[1]
    Results_Sweep(filename)