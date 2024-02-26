from Sort_Data_Infinite import Sort
from Transfer_Matrix_Model_Solver import Solve
from Post_Process import default_Calculations
import time
import numpy as np
import json
from Coupled_Wave_Solver import Coupled_Wave_Solver_EEDFB
from Post_Process import index_Plot
class NumpyEncoder(json.JSONEncoder):
    ### Custom encoder for numpy data types 
    ### https://stackoverflow.com/a/47626762/19873191 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)

def load_parameters(json_filepath):
    ### Converting from list to array recursively:
    def convert_to_array(item):
        if isinstance(item, list):
            try:
                return np.array(item)
            except ValueError:
                return np.array([convert_to_array(subitem) for subitem in item])
        return item
    """Load parameters from a JSON file and convert specific lists to numpy arrays."""
    with open(json_filepath, 'r') as file:
        params = json.load(file)

    ### These will be converted from lists back to nd.array which makes my life easier!!!!!
    ### Mainly just the results from Solve()
    array_keys_top_level = ['alpha_m_all', 'delta_m_all', 'Gamma_lg_all', 'alpha_surf_all', 'Ratio_all', 'Guided_export_all', 'R_export_all', 'S_export_all', 'P_R_all', 'P_L_all', 'alpha_end_all', 'alpha_end_R_all', 'alpha_end_L_all']
    
    for key in array_keys_top_level:
        if key in params:  # Check if the key exists at the top level
            params[key] = convert_to_array(params[key])
    return params

def save_parameters(params, json_filepath):
    ### Save parameters, including numpy arrays, to a JSON file.
    with open(json_filepath, 'w') as file:
        json.dump(params, file, indent=4, cls=NumpyEncoder)

def EEDFB_Solver(json_filepath, params_sweep = False):
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
    params['Lambda'] = Lambda
    L = params['L']  # DFB region cm
    l = params['l']  # DBR region cm
    L_trans = params['L_trans']
    rR = params['rR']  # front facet reflectivity r - AR-coated = 0.372, Uncoated = 0.5196
    rL = params['rL']  # back facet reflectivity r
    num_Z = params['num_Z'] # number of points in Z to calculate field profiles at 
    ### Indices of parameters i, j, k to sweep. Script can be updated for N=4+ parametric sweeps, but will need some work since direct indexing is used.
    params_indices = params['params_indices']
    # plot_SWEEP = 1 ### This is whether to plot during DFB_Sweep the mode spectrum. Should be set to 0 when not testing, which is the default in the JSON template file.
    plot_SWEEP = params['plot_SWEEP']
    ### Initialized to 100, but the script will effectively never generate 100
    num_Modes = params['num_Modes']
    ### I don't know what this is lol, but it's used constantly in the original files, so I'll include it here
    Gamma_ele = params['Gamma_ele']
    ### Phase shift due to cleave position at end of cavities
    cleave_phase_shift = params['cleave_phase_shift']
    ### Phase shift at center of DFB cavity
    pi_phase_shift = params['pi_phase_shift']
    ### If there is a DBR, there will be some reflectivity between the DBR and DFB sections.
    ### If not using a DBR, set this to 0.
    r_DFB_DBR = params['r_DFB_DBR']
    ### Whether to plot the profiles in Coupled Wave Solver
    plot_fields = params['plot_fields']
    ### Plot of index values
    plot_Indices = params['plot_Indices']
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if not modes_solved:
        derived_values = Sort(filename, wavelength, Lambda)
        params['cladding_thickness'] = derived_values['params'][0]
        params['grating_height'] = derived_values['params'][1]
        params['duty_cycle'] = derived_values['params'][2]
            ### Moving to plotting
        if plot_Indices:
            index_Plot(params['cladding_thickness'], params['grating_height'])
    
    
        alpha_m_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        delta_m_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        Gamma_lg_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        alpha_surf_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        Ratio_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        kappa_DFB_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0]))
        
        Guided_export_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes, num_Z)) 
        R_export_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes, num_Z))  
        S_export_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes, num_Z))  
        
        P_R_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        P_L_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        alpha_end_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        alpha_end_R_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        alpha_end_L_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        z_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes, num_Z))  

        start = time.time()

        for i in range(params_indices[0][0], params_indices[0][1]):
            for j in range(params_indices[1][0], params_indices[1][1]):
                for k in range(params_indices[2][0], params_indices[2][1]):
                    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                    #### Transfer Matrix Method for EEDFB - finds modes w/ alpha_m (used for gth), detuning (finds wavelength), coupling (import parameter), 
                    #### coupling parameter (counterpart to Kappa)
                    # (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve(i, j, k, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z)
                    (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve(i, j, k, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
                    ### Alpha_m contains the number of modes found
                    ### Unused entries in _all are set to NaN for easier post-processing.
                    
                    (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z) = Coupled_Wave_Solver_EEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                              Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                              cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields)
                    

                    alpha_m_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_m)] = alpha_m 
                    alpha_m_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(alpha_m):] = np.NaN 
                    delta_m_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(delta_m)] = delta_m
                    delta_m_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(delta_m):] = np.NaN 
                    Gamma_lg_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(Gamma_lg)] = Gamma_lg
                    Gamma_lg_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(Gamma_lg):] = np.NaN  
                    alpha_surf_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_surf)] = alpha_surf
                    alpha_surf_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(alpha_surf):] = np.NaN 

                    Guided_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(Guided_export)] = Guided_export
                    Guided_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(Guided_export):] = np.NaN  
                    R_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0],:len(R_export)] = R_export
                    R_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(R_export):] = np.NaN  
                    S_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(S_export)] = S_export
                    S_export_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(S_export):] = np.NaN  
                    
                    P_R_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(P_R)] = P_R
                    P_R_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0],len(P_R):] = np.NaN  
                    P_L_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(P_L)] = P_L
                    P_L_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(P_L):] = np.NaN  
                    alpha_end_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_end)] = alpha_end
                    alpha_end_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(alpha_end):] = np.NaN
                    alpha_end_R_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_end_R)] = alpha_end_R
                    alpha_end_R_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], len(alpha_end_R):] = np.NaN
                    alpha_end_L_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_end_L)] = alpha_end_L
                    alpha_end_L_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0],len(alpha_end_L):] = np.NaN
                    
                    z_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(z)] = z
                    z_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0],len(z):] = z 
                    kappa_abs = np.abs(derived_values['kappa_DFB'][i,j,k])
                    kappa_DFB_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0]] = kappa_abs
                    
                    
        print(f"Mode solver time elapsed is {time.time() - start} seconds.") 
        params['alpha_m_all'] = alpha_m_all
        params['delta_m_all'] = delta_m_all
        params['Gamma_lg_all'] = Gamma_lg_all
        params['alpha_surf_all'] = alpha_surf_all
        params['Guided_export_all'] = Guided_export_all
        params['R_export_all'] = R_export_all
        params['S_export_all'] = S_export_all
        params['P_R_all'] = P_R_all
        params['P_L_all'] = P_L_all
        params['alpha_end_all'] = alpha_end_all
        params['alpha_end_R_all'] = alpha_end_R_all
        params['alpha_end_L_all'] = alpha_end_L_all
        params['z_all'] = z_all
        params['kappa_DFB'] = kappa_DFB_all
        ### Store in params file. If modes_solved = 1, skip the Sort_Data_Infinite and DFB_Sweep steps
        ## Export to JSON with modes_solved flag set to 1. Then, on re-execution of the same JSON config file, the solutions will be automatically loaded, i.e. moving to Plot_DFB. 
        modes_solved = 1
        params['modes_solved'] = modes_solved
        
        if not params_sweep:
            save_parameters(params, json_filepath)
        else:
            print(f"Moving to post-processing")
            results = default_Calculations(params, params_sweep)
            return results

    ### Post-processing 
    ### If params_sweep, this should be hit -> we're not saving the results file for each value in the param sweep (datasize would blow up for large sweeps)
    ### so the data is dropped after the `return results` line above. If results_sweep (i.e. not running a new simulation each execution), the previous sim is 
    ### reused (so if modes_solved = 1, params_sweep = 0), we move here. 
    if modes_solved and not params_sweep:    
        print(f"Moving to post-processing")
        params = load_parameters(json_filepath)
        #print(f"Testing: alpha_m type: {type(Data['alpha_m_all'])}")
        #print(f"Testing: sub-lists of alpha_m type: {type(Data['alpha_m_all'][0])}")
        
        ### The 'results indices' parameter is used for the mode spectrum/field profile results. 
        ### By default, the entire sweep is plotted in the contour plots.
        ### I could probably update this later to let uesrs choose the range for the contours.
        ### We can use return here as well to collect collected results for further post-processing 
        ### or inputting into the Machine Learning pipeline.
        ### One option would be automating the full pipeline [essentially complete w/ wrapper], 
        ### generating waveguide characteristics, and feeding these into K.P/NextNano ML scripts.
        results = default_Calculations(params, params_sweep)
    return results
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    
    ### Loading parameters. This helps keep us from overwriting configurations for a run.
    ### All you really need to change is the JSON filepath. Edit parameters within the JSON file so that they're not overwritten for future runs. 
    ### If internal parameters need to be changed, just update the JSON file, and load them from there. Then, update the script so that they're passed into the solver.
#json_filepath = r"Data/DRS_4.9um_EEDFB_2-23-2024_3mm.json"
json_filepath = r"Data/9.9um_Design_BH_DFB_02-23-2024.json"
EEDFB_Solver(json_filepath)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    


### Order of operations ->
### Run EEDFB_Solver initially to find the correct index values (set plot_Indices to 1 in the JSON file)
### Either let the execution complete, and move to results_Sweep to pull out explicit values into the excel file - take note of the indices figure
### Otherwise, if varying a non-trivial parameter (e.g. reflectivity, cleave position, etc.) cancel the execution and run param_Sweep. param_Sweep will run for a *very* long time, so be careful.
