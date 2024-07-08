from SortCSV import SortCSV
from Transfer_Matrix_Model_Solver_CSV import Solve_EE, Solve_SE
from Post_Process_EE_NP import default_Calculations
import numpy as np
import json
from Coupled_Wave_Solver_NP import Coupled_Wave_Solver_EEDFB, Coupled_Wave_Solver_SEDFB
import os
from pickle_utils import read_pickle, write_pickle, add_metadata, save_sort

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Please note that this requires the output to be in .csv form


class NumpyEncoder(json.JSONEncoder):
    ### Custom encoder for numpy data types 
    ### https://stackoverflow.com/a/47626762/19873191 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

# Loading the JSON file
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
    # Obsolete, leaving this in until the script is finished
    array_keys_top_level = ['alpha_m', 'delta_m', 'Gamma_lg', 'alpha_surf', 'Ratio', 'Guided_export', 'R_export', 'S_export', 'P_R', 'P_L', 'alpha_end', 'alpha_end_R', 'alpha_end_L']
    for key in array_keys_top_level:
        if key in params:  # Check if the key exists at the top level
            params[key] = convert_to_array(params[key])
    return params

# Writing the JSON file
def save_parameters(params, json_filepath):
    ### Save parameters, including numpy arrays, to a JSON file.
    with open(json_filepath, 'w') as file:
        json.dump(params, file, indent=4, cls=NumpyEncoder)

# Solver script. Analagous to 'DFB_Sweep_2.mat'
def DFB_Solver(json_filepath, params_sweep=False):
    params = load_parameters(json_filepath)
    filename = params['filename']
    results_fn = "Data/" + os.path.splitext(filename)[0] + '_' + str(np.datetime64('today')) + '.pkl'
    filename = "Data/" + filename


    ### Storage container for all data.
    params = load_parameters(json_filepath)
    ### Number of parametric variations
    ND = params['ND']
    ### If Adaptive Mesh Refinement was used. If true, picks the final ARM solution and removes the others for each parametric point.
    AMR = params['AMR']
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

    # For processing raw data from CSV. 
    headers = {
        'gamma_col': params.get('gamma_col', 'dom2/dom3, Global Var Gamma'),
        'delta_kappa_col': params.get('delta_kappa_col', '(freq*2*pi/c_const-2*pi/wavelength[um])*wavelength/Period/2, delta kappa'),
        'loss_col': params.get('loss_col', 'emw.damp/emw.freq/(Period[um]), Global Var Loss'),
        'wavelength_col': params.get('wavelength_col', 'c_const/emw.freq, Global Var wavelength'),
        'surf_loss_col': params.get('surf_loss_col', 'bnd1/dom1, Global Variable Surface Loss'),
        'sorting_col_name': params.get('sorting_col_name', 'gamma_col') # Set this to the key corresponding to the column to be used
    }


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if not modes_solved:

        SortCSV(filename, results_fn, wavelength, Lambda, ND, AMR)

        # Loading the pickle file that was saved in SortCSV
        data = read_pickle(results_fn)
        parameters, inputs = data['parameters'], data['inputs']
        num_rows = len(next(iter(parameters.values()))) # Essentially grabbing the row count from the dict.

        results = {
            'alpha_m': [],
            'delta_m': [],
            'Gamma_lg': [],
            'alpha_surf': [],
            'Guided_export': [],
            'R_export': [],
            'S_export': [],
            'P_R': [],
            'P_L': [],
            'alpha_end': [],
            'alpha_end_R': [],
            'alpha_end_L': [],
            'z': []
        }
        
        print(f"There are a total of {num_rows} unique simulation points.")

        for row in range(num_rows):
            row_parameters = {key: parameters[key][row] for key in parameters}
            row_inputs = {key: inputs[key][row] for key in inputs}
            
            DC = row_inputs.get('DC') or row_inputs.get('DutyCycle') or 0.5
            
            # These are derived in the 'SortCSV' script. They are based off of the solutions to the infinite waveguide model. 
            derived_values = {}
            derived_values = {
                'deltak_DFB': [row_inputs['deltak1_DFB'], row_inputs['deltak2_DFB']],
                'deltak_DBR': [row_inputs['deltak1_DBR'], row_inputs['deltak2_DBR']],
                'alpha_DFB': [row_inputs['alpha1_DFB'], row_inputs['alpha2_DFB']],
                'alpha_DBR': [row_inputs['alpha1_DBR'], row_inputs['alpha2_DBR']],
                'asurf_DFB': row_inputs['asurf_DFB'],
                'asurf_DBR': row_inputs['asurf_DBR'],
                'kappa_DFB': row_inputs['kappa_DFB'],
                'kappa_DBR': row_inputs['kappa_DBR'],
                'zeta_DFB': row_inputs['zeta_DFB'],
                'zeta_DBR': row_inputs['zeta_DBR']
            }
            
            if SEDFB: # - Needs to be refactored
                # ### Easier to use the imaginary part as its own row
                # asurf_trans_ = np.loadtxt("Data/asurf_trans.csv", delimiter=',')
                # #asurf_trans = asurf_trans_[:, 0] + 1j * asurf_trans_[:, 1]
                # alpha_trans_ = np.loadtxt("Data/alpha_trans.csv", delimiter=',')
                # #alpha_trans = alpha_trans_[:, 0] + 1j * alpha_trans_[:, 1]
                # deltak_trans_ = np.loadtxt("Data/deltak_trans.csv", delimiter=',')
                # #deltak_trans = deltak_trans_[:, 0] + 1j * deltak_trans_[:, 1]
                # k0 = 2 * np.pi / wavelength
                # K0 = np.pi / Lambda

                # kappa_trans = -(deltak_trans_[..., 0] - deltak_trans_[..., 1]) * K0 / 2 / k0 + 1j * (alpha_trans_[..., 0] - alpha_trans_[..., 1]) / 2
                # zeta_trans = -(deltak_trans_[..., 0] + deltak_trans_[..., 1]) * K0 / 2 / k0 + 1j * (alpha_trans_[..., 0] + alpha_trans_[..., 1]) / 2
                
                # (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve_SE(i_p, j_p, k_p, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, plot_SWEEP, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
            
                # (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z) = Coupled_Wave_Solver_SEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                #         Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                #         cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields, kappa_trans, zeta_trans, asurf_trans_[0])            
                continue
            #### EE DFB Calculations
            else:                                                                                           #(L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, Plot_SWEEP, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
                (alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR) = Solve_EE(DC, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
            
                print(f"Number of modes found for solution {row + 1} of {num_rows}: {len(alpha_m)}")

                (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z) = Coupled_Wave_Solver_EEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                        Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                        cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields)
                
                def array_to_list(arr):
                    if isinstance(arr, np.ndarray):
                        return [array_to_list(item) for item in arr]
                    elif isinstance(arr, list):
                        return [array_to_list(item) for item in arr]
                    else:
                        return arr

                results['alpha_m'].append(alpha_m)
                results['delta_m'].append(delta_m)
                results['Gamma_lg'].append(Gamma_lg)
                results['alpha_surf'].append(alpha_surf)
                results['Guided_export'].append(Guided_export)
                results['R_export'].append(R_export)
                results['S_export'].append(S_export)
                results['P_R'].append(P_R)
                results['P_L'].append(P_L)
                results['alpha_end'].append(alpha_end)
                results['alpha_end_R'].append(alpha_end_R)
                results['alpha_end_L'].append(alpha_end_L)
                results['z'].append(z)
        
        params['modes_solved'] = 1

        
        if not params_sweep:
            data['results'] = results
            write_pickle(results_fn, data)
            add_metadata(results_fn, {'filename': filename, 'json_filepath': json_filepath})
            params['results_fn'] = results_fn
            save_parameters(params, json_filepath)
            
        else:
            # Needs to be updated
            #print(f"Moving to post-processing")
            #results = default_Calculations(params, params_sweep)
            #return results
            pass        

    if params['modes_solved'] and not params_sweep:
        print(f"Moving to post-processing")
        params = load_parameters(json_filepath)
        data = read_pickle(results_fn)

        output = default_Calculations(params, data)
        
        
    return output

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@    
    # The JSON file contains constant inputs corresponding to the simulation data.
    # The pickle file contains all simulation results.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@

# If you want to use CLI or the param_Sweep, make sure to comment out the explicit file declarations. Otherwise, param_sweep will initialize
# by running the script *for that file*, which is generally unintended.
if __name__ == "__main__":
    #filename = sys.argv[1]
    
    filename = 'EE_NP_JSON_template.json'
    DFB_Solver("Data/" + str(filename))

