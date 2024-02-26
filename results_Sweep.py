from EEDFB_Finite_Mode_Solver import EEDFB_Solver
from EEDFB_Finite_Mode_Solver import load_parameters
from EEDFB_Finite_Mode_Solver import save_parameters
from Post_Process import default_Calculations
import json
import os
import pandas as pd
import time
from Post_Process import index_Plot
### This is explicitly for varying a trivial parameters (so duty cycle, cladding thickness, or grating height)
### Best method is to run EEDFB_Finite_Mode_Solver.py first, find the indices to target, and then run this to grab the csv file export.
filename_original = r"9.9um_Design_BH_DFB_02-23-2024.json"

excel_out, _ = os.path.splitext(filename_original)
excel_out ="Data/" + excel_out + ".xlsx"


### 2nd index is exclusive - i.e. if you want 0 : 5, use 0, 6
sweep_indices = [
    range(0, 17),  
    range(3, 5),   
    range(1, 2)    
]

headers = ['rR', 'duty_cycle', 'cladding_thickness', 
           'grating_height', 'PmaxWPE_est_CW', 'kappa_DFB_L', 'S0gacterm', 
           'AReff', 'Homega_4', 'alpha_m', 'Jth', 'del_gth', 'eta_s_est', 
           'tau_photon', 'tau_stim_4', 'Field_end'
]
data = {header: [] for header in headers}
params = load_parameters("Data/" + filename_original)
if params['plot_Indices']:
    index_Plot(params['cladding_thickness'], params['grating_height'])
    
for i in sweep_indices[0]:
    for j in sweep_indices[1]:
        for k in sweep_indices[2]:
            start = time.time()
            print(i)
            ### Setting results_indices for this iteration
            results_indices = [i, j, k]
            params['results_indices'] = results_indices
            
            if params['modes_solved']:
                results = default_Calculations(params, False)
            else:
                save_parameters(params, "Data/" + filename_original)
                results = EEDFB_Solver("Data/" + filename_original)
                params = load_parameters("Data/" + filename_original)
                                   
            for header, result in zip(headers, results):
                data[header].append(result)
            print(f"Mode solver time elapsed is {time.time() - start} seconds.") 


df = pd.DataFrame(data)
df.to_excel(excel_out, index=False)
