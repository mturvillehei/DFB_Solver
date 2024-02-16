from Sort_Data_Infinite import Sort
from Transfer_Matrix_Model_Solver import Solve
import time
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    ### Custom encoder for numpy data types 
    ### https://stackoverflow.com/a/47626762/19873191 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)

def load_parameters(json_filepath):
    """Load parameters from a JSON file and convert specific lists to numpy arrays."""
    with open(json_filepath, 'r') as file:
        params = json.load(file)
    
    ### These will be converted from lists back to nd.array which makes my life easier!!!!!
    ### Mainly just the results from Solve()
    array_keys_top_level = ['delta_m_all', 'Gamma_lg_all', 'alpha_surf_all', 'Ratio_all', 'Guided_export_all', 'R_export_all', 'S_export_all', 'P_R_all', 'P_L_all']
    
    for key in array_keys_top_level:
        if key in params:  # Check if the key exists at the top level
            params[key] = np.array(params[key])
    
    return params

def save_parameters(params, json_filepath):
    ### Save parameters, including numpy arrays, to a JSON file.
    with open(json_filepath, 'w') as file:
        json.dump(params, file, indent=4, cls=NumpyEncoder)

def Main(json_filepath):
    
   
    params = load_parameters(json_filepath)
    filename = params['filename']
    filename = "Data/" + filename
    
    ### Storage container for all data.
    params = load_parameters(json_filepath)
    ### Boolean that checks if Solve() has already been performed - initialized to 0. After sweep, set to 1 to allow for post-processing.
    modes_solved = params['modes_solved']        
    ### fundamental mode wavelength 
    wavelength = params['wavelength']
    ### neff guess, used for approximating Lambda (essentially no dependence except for the number of periods calculated)
    neff = params['neff']
    ### Grating period
    Lambda = wavelength / (2 * neff)
    L = params['L']  # DFB region mm
    l = params['l']  # DBR region mm
    rR = params['rR']  # front facet reflectivity r
    rL = params['rL']  # back facet reflectivity r
    num_Z = params['num_Z'] # number of points in Z to calculate field profiles at 
    ### Indices of parameters i, j, k to sweep. Script can be updated for N=4+ parametric sweeps, but will need some work since direct indexing is used.
    params_indices = params['params_indices']
    # plot_SWEEP = 1 ### This is whether to plot during DFB_Sweep the mode spectrum. Should be set to 0 when not testing, which is the default in the JSON template file.
    plot_SWEEP = params['plot_SWEEP']
    ### Initialized to 100, but the script will effectively never generate 100
    num_Modes = params['num_Modes']

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Placed the above in a JSON conf file  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if not modes_solved:
        #print(params_indices)
        derived_values = Sort(filename, wavelength, Lambda)
        #print(derived_values.keys())

        ### DFB Sweep ###

        alpha_m_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        delta_m_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        Gamma_lg_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        alpha_surf_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        Ratio_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        
        Guided_export_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes, num_Z)) 
        R_export_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes, num_Z)) 
        S_export_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes, num_Z)) 
        
        P_R_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
        P_L_all = np.zeros((len(params_indices[0]), len(params_indices[1]), len(params_indices[2]), num_Modes))
               
        
        for i in params_indices[0]:
            for j in params_indices[1]:
                for k in params_indices[2]:
                    
                    start = time.time()
                    #print(f"Kappa for x1 = {params[0][i]}, x2 = {params[1][j]}, x3 = {params[2][k]} is {derived_values['kappa'][i][j][k]}")
                    (alpha_m, delta_m, Gamma_lg, alpha_surf, Ratio, Guided_export, R_export, S_export, P_R, P_L) = Solve(i, j, k, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z)
                    print(f"Time elapsed is {time.time() - start} seconds.")
                    ### Alpha_m contains the number of modes found
                    modes_found = len(alpha_m)
                    print(f"Modes found: {modes_found}")
                    ### Unused entries in _all are set to NaN for easier post-processing.
                    alpha_m_all[i, j, k, :len(alpha_m)] = alpha_m 
                    alpha_m_all[i, j, k, len(alpha_m):] = np.NaN 
                    delta_m_all[i, j, k, :len(delta_m)] = delta_m
                    delta_m_all[i, j, k, len(delta_m):] = np.NaN 
                    Gamma_lg_all[i, j, k, :len(Gamma_lg)] = Gamma_lg
                    Gamma_lg_all[i, j, k, len(Gamma_lg):] = np.NaN  
                    alpha_surf_all[i, j, k, :len(alpha_surf)] = alpha_surf
                    alpha_surf_all[i, j, k, len(alpha_surf):] = np.NaN 
                    Ratio_all[i, j, k, :len(Ratio)] = Ratio
                    Ratio_all[i, j, k, len(Ratio):] = np.NaN  #
                    
                    Guided_export_all[i, j, k, :len(Guided_export)] = Guided_export
                    Guided_export_all[i, j, k, len(Guided_export):] = np.NaN  
                    R_export_all[i, j, k, :len(R_export)] = R_export
                    R_export_all[i, j, k, len(R_export):] = np.NaN  
                    S_export_all[i, j, k, :len(S_export)] = S_export
                    S_export_all[i, j, k, len(S_export):] = np.NaN  
                    
                    P_R_all[i, j, k, :len(P_R)] = P_R
                    P_R_all[i, j, k, len(P_R):] = np.NaN  
                    P_L_all[i, j, k, :len(P_L)] = P_L
                    P_L_all[i, j, k, len(P_L):] = np.NaN  
                    
        params['delta_m_all'] = delta_m_all
        params['Gamma_lg_all'] = Gamma_lg_all
        params['alpha_surf_all'] = alpha_surf_all
        params['Ratio_all'] = Ratio_all
        params['Guided_export_all'] = Guided_export_all
        params['R_export_all'] = R_export_all
        params['S_export_all'] = S_export_all
        params['P_R_all'] = P_R_all
        params['P_L_all'] = P_L_all
                    
        ### Store in params file. If modes_solved = 1, skip the Sort_Data_Infinite and DFB_Sweep steps
        ## Export to JSON with modes_solved flag set to 1. Then, on re-execution of the same JSON config file, the solutions will be automatically loaded, i.e. moving to Plot_DFB. 
        params['modes_solved'] = 1
        save_parameters(params, json_filepath)
    
    ### To prevent overlap. Probably not needed but hey, why not.
    del params
    
    ### Post-processing 
    if modes_solved:    
        
        Data = load_parameters(json_filepath)
        
        print(f"Fill with post-processing")
        
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    
    ### Loading parameters. This helps keep us from overwriting configurations for a run.
    ### All you really need to change is the JSON filepath. Edit parameters within the JSON file so that they're not overwritten for future runs. 
    ### If internal parameters need to be changed, just update the JSON file, and load them from there. Then, update the script so that they're passed into the solver.
    
json_filepath = r"Data/5L_Test_File_2.json"
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    

Main(json_filepath)