from Sort_Data_Infinite import Sort
from Transfer_Matrix_Model_Solver import Solve_EE, Solve_SE
from Post_Process_EE import default_Calculations
import time
import numpy as np
import json
from Coupled_Wave_Solver import Coupled_Wave_Solver_EEDFB, Coupled_Wave_Solver_SEDFB
from Post_Process_EE import index_Plot
import sys

class NumpyEncoder(json.JSONEncoder):
    ### Custom encoder for numpy data types 
    ### https://stackoverflow.com/a/47626762/19873191 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def find_Args(params, key):
    i_params = []
    j_params = []
    k_params = []

    ### Ensuring the endpoint is included
    increment = 1e-5
    i_ps = np.arange(params[key][0][0], params[key][0][1] + params[key][0][2] * increment, params[key][0][2])
    j_ps = np.arange(params[key][1][0], params[key][1][1] + params[key][1][2] * increment, params[key][1][2])
    k_ps = np.arange(params[key][2][0], params[key][2][1] + params[key][2][2] * increment, params[key][2][2])
    
    for i_p in i_ps:
        differences = np.abs(params['cladding_thickness'] - i_p)    
        i_c = np.argmin(differences)  
        ### If error is more than floating point
        if params['cladding_thickness'][i_c] - i_p > 0.001:
            raise Exception(f"for index {i_c} in cladding thickness, the best match is {params['cladding_thickness'][i_c]}, which does not meet the precision requirement.")
        else:
            i_params.append(i_c)           
    for j_p in j_ps:   
        differences = np.abs(params['grating_height'] - j_p)    
        j_c = np.argmin(differences)  
        ### If error is more than floating point
        if params['grating_height'][j_c] - j_p > 0.001:
            raise Exception(f"for index {j_c} in grating_height, the best match is {params['grating_height'][j_c]}, which does not meet the precision requirement.")
        else:
            j_params.append(j_c)
    for k_p in k_ps:    
        differences = np.abs(params['duty_cycle'] - k_p)    
        k_c = np.argmin(differences)  
        ### If error is more than floating point
        if params['duty_cycle'][k_c] - k_p > 0.001:
            raise Exception(f"for target {k_p} in duty cycle, the best match is {params['duty_cycle'][k_c]}, which does not meet the precision requirement.")
        else:
            k_params.append(k_c)
    return i_params, j_params, k_params

def load_parameters(json_filepath):
    ### Converting from list to array recursively:
    def convert_to_array(item):
        if isinstance(item, list):
            try:
                return np.array(item)
            except ValueError:
                return np.array([convert_to_array(subitem) for subitem in item])
        return item
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

def DFB_Solver(json_filepath, params_sweep = False):
    params = load_parameters(json_filepath)
    filename = params['filename']
    filename = "Data/" + filename
    
    ### Storage container for all data.
    params = load_parameters(json_filepath)
    ### Boolean that checks if Solve() has already been performed - initialized to 0. After sweep, set to 1 to allow for post-processing.
    modes_solved = params['modes_solved']        
    ### Identifying if the device is EE or SE. 1 for SE, 0 for EE.
    SEDFB = params['SEDFB']
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


        ### Extrapolating the correct array indices 
        i_params, j_params, k_params = find_Args(params, 'params_values')
        params['params_indices'] = [i_params, j_params, k_params]
        
    
        #alpha_m_all = np.zeros((params_indices[0][1] - params_indices[0][0], params_indices[1][1] - params_indices[1][0], params_indices[2][1] - params_indices[2][0], num_Modes))
        alpha_m_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        
        delta_m_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        Gamma_lg_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        alpha_surf_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        Ratio_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        kappa_DFB_all = np.zeros((len(i_params), len(j_params), len(k_params)))
        
        Guided_export_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes, num_Z)) 
        R_export_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes, num_Z))  
        S_export_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes, num_Z))  
        
        P_R_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        P_L_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        alpha_end_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        alpha_end_R_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        alpha_end_L_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes))
        z_all = np.zeros((len(i_params), len(j_params), len(k_params), num_Modes, num_Z))  

        start = time.time()

        for i in range(len(i_params)):
            for j in range(len(j_params)):
                for k in range(len(k_params)):
                    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                    #### Transfer Matrix Method for EEDFB - finds modes w/ alpha_m (used for gth), detuning (finds wavelength), coupling (import parameter), 
                    #### coupling parameter (counterpart to Kappa)
                    solver_time = time.time()

                    i_p = i_params[i]
                    j_p = j_params[j]
                    k_p = k_params[k]
                    
                    
                    #### SE DFB Calculations
                    if SEDFB:
                        ### Easier to use the imaginary part as its own row
                        asurf_trans_ = np.loadtxt("Data/asurf_trans.csv", delimiter=',')
                        #asurf_trans = asurf_trans_[:, 0] + 1j * asurf_trans_[:, 1]
                        alpha_trans_ = np.loadtxt("Data/alpha_trans.csv", delimiter=',')
                        #alpha_trans = alpha_trans_[:, 0] + 1j * alpha_trans_[:, 1]
                        deltak_trans_ = np.loadtxt("Data/deltak_trans.csv", delimiter=',')
                        #deltak_trans = deltak_trans_[:, 0] + 1j * deltak_trans_[:, 1]
                        k0 = 2 * np.pi / wavelength
                        K0 = np.pi / Lambda

                        kappa_trans = -(deltak_trans_[..., 0] - deltak_trans_[..., 1]) * K0 / 2 / k0 + 1j * (alpha_trans_[..., 0] - alpha_trans_[..., 1]) / 2
                        zeta_trans = -(deltak_trans_[..., 0] + deltak_trans_[..., 1]) * K0 / 2 / k0 + 1j * (alpha_trans_[..., 0] + alpha_trans_[..., 1]) / 2
                        
                        (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve_SE(i_p, j_p, k_p, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
                    
                        (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z) = Coupled_Wave_Solver_SEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                              Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                              cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields, kappa_trans, zeta_trans, asurf_trans_[0])            
                        
                        
                    else:
                        (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve_EE(i_p, j_p, k_p, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
                    
                        (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z) = Coupled_Wave_Solver_EEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                              Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                              cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields)
                        
                    ### Alpha_m contains the number of modes found
                    ### Unused entries in _all are set to NaN for easier post-processing.
                    
                    print(f"Mode solver time elapsed is {time.time() - solver_time} seconds.") 

                    #alpha_m_all[i - params_indices[0][0], j - params_indices[1][0], k - params_indices[2][0], :len(alpha_m)] = alpha_m 
                    alpha_m_all[i, j, k, :len(alpha_m)] = alpha_m 
                    
                    alpha_m_all[i, j, k, len(alpha_m):] = np.NaN 
                    delta_m_all[i, j, k, :len(delta_m)] = delta_m
                    delta_m_all[i, j, k, len(delta_m):] = np.NaN 
                    Gamma_lg_all[i, j, k, :len(Gamma_lg)] = Gamma_lg
                    Gamma_lg_all[i, j, k, len(Gamma_lg):] = np.NaN  
                    alpha_surf_all[i, j, k, :len(alpha_surf)] = alpha_surf
                    alpha_surf_all[i, j, k, len(alpha_surf):] = np.NaN 

                    Guided_export_all[i, j, k, :len(Guided_export)] = Guided_export
                    Guided_export_all[i, j, k, len(Guided_export):] = np.NaN  
                    R_export_all[i, j, k,:len(R_export)] = R_export
                    R_export_all[i, j, k, len(R_export):] = np.NaN  
                    S_export_all[i, j, k, :len(S_export)] = S_export
                    S_export_all[i, j, k, len(S_export):] = np.NaN  
                    
                    P_R_all[i, j, k, :len(P_R)] = P_R
                    P_R_all[i, j, k,len(P_R):] = np.NaN  
                    P_L_all[i, j, k, :len(P_L)] = P_L
                    P_L_all[i, j, k, len(P_L):] = np.NaN  
                    alpha_end_all[i, j, k, :len(alpha_end)] = alpha_end
                    alpha_end_all[i, j, k, len(alpha_end):] = np.NaN
                    alpha_end_R_all[i, j, k, :len(alpha_end_R)] = alpha_end_R
                    alpha_end_R_all[i, j, k, len(alpha_end_R):] = np.NaN
                    alpha_end_L_all[i, j, k, :len(alpha_end_L)] = alpha_end_L
                    alpha_end_L_all[i, j, k,len(alpha_end_L):] = np.NaN
                    
                    z_all[i, j, k, :len(z)] = z
                    z_all[i, j, k,len(z):] = z 
                    kappa_abs = np.abs(derived_values['kappa_DFB'][i,j,k])
                    kappa_DFB_all[i, j, k] = kappa_abs
                    
                    
        print(f"Full time elapsed is {time.time() - start} seconds.") 
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
    ### reused, we move here. 
    
    if modes_solved and not params_sweep:    
        print(f"Moving to post-processing")
        params = load_parameters(json_filepath)
        ### The 'results indices' parameter is used for the mode spectrum/field profile results. 
        ### By default, the entire sweep is plotted in the contour plots.

        ### One option would be automating the full pipeline [essentially complete w/ wrapper], 
        ### generating waveguide characteristics, and feeding these into K.P/NextNano ML scripts.
        
        results = default_Calculations(params, params_sweep)
    return results
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    
    ### Loading parameters. This helps keep us from overwriting configurations for a run.
    ### All you really need to change is the JSON filepath. Edit parameters within the JSON file so that they're not overwritten for future runs. 
    ### If internal parameters need to be changed, just update the JSON file, and load them from there. Then, update the script so that they're passed into the solver.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    

### If you want to use CLI or the param_Sweep, make sure to comment out the explicit file declarations.
if __name__ == "__main__":
    #filename = sys.argv[1]
    #filename = "9.9um_60stg_Test_Against_Matlab.json"
    filename = "DRS_4.9um_EEDFB_2-23-2024.json"
    DFB_Solver("Data/" + str(filename))
### Run EEDFB_Solver initially to find the correct index values (set plot_Indices to 1 in the JSON file). Can be run from CLI by commenting out line 287 for 286
### Either let the execution complete, and move to results_Sweep
### Otherwise, if varying a non-trivial parameter (e.g. reflectivity, cleave position, etc.) cancel the execution and run param_Sweep. param_Sweep will run for a *very* long time, so be careful.
