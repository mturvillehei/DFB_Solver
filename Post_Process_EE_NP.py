import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pickle_utils import read_pickle, save_post_process
import pandas as pd


def default_Calculations(JSONData, pickle_data):
    
    # Preset parameters. Can be varied in the param sweep.
    L = JSONData['L'] ### cm used here, not mm
    f = JSONData['f'] ### frequency for h(w) calculation
    Gamma = JSONData['Gamma']
    alpha_w = JSONData['alpha_w'] ### Waveguide loss. Calculated from 2D cross-section simulation.
    alpha_bf = JSONData['alpha_bf'] ### backfilling loss,  assumed to be 0.1 by Dan
    Jmax = JSONData['Jmax'] ### Peak current 
    JmaxWPE = JSONData['JmaxWPE'] ### Jmax at peak WPE 
    g = JSONData['g'] ### Differential gain. See IFR_Differential_Gain (email me if you need it)
    wavelength = JSONData['wavelength'] ### wavelength 
    Lambda = JSONData['Lambda'] ### grating length
    ni = JSONData['ni'] ### internal efficiency
    NP = JSONData['NP'] ### number of stages
    F1 = JSONData['F1'] ### used for calculating slope efficiency. not sure where this is from 
    HR = JSONData['HR'] ### HR percent on the back facet ### little r or big r?
    q = JSONData['q'] ### charge unit
    h = JSONData['h'] ### planck's
    c = JSONData['c'] ### speed of light
    P = JSONData['P'] ### power (for calculating circulating power)
    Wid = JSONData['Wid'] ### device width
    Stage_t = JSONData['stage_t'] ### stage thickness
    tau_up = JSONData['tau_up'] ### tau_upper state -> tau,ul,global [IFR + AD + LO]
    #tau_32 = JSONData['tau_32'] ### tau_3->2 lifetime
    tau_32 = tau_up
    neff = JSONData['neff']
    CW_Scaling_Factor = JSONData['CW_Scaling_Factor']
    
    # 2D dataframe. This can be optimized by an optimizer
    scalar_column_names = [
        'L', 'f', 'Gamma', 'alpha_w', 'alpha_bf', 'Jmax', 'JmaxWPE', 'g', 'wavelength', 'Lambda',
        'ni', 'NP', 'F1', 'HR', 'q', 'h', 'c', 'P', 'Wid', 'Stage_t', 'tau_up', 'tau_32', 'neff',
        'CW_Scaling_Factor', 'kappa_DFB', 'alpha_m', 'delta_m', 'Gamma_lg', 'alpha_surf', 'P_R',
        'P_L', 'alpha_end', 'alpha_end_L', 'alpha_end_R', 'alpha_i', 'alpha_opt', 'k0', 'K0',
        'energy', 'AReff', 'nu', 'dAR', 'Lp', 'g_diffunit', 'gc', 'sg', 'gth', 'I', 'F', 'Jth',
        'lossterm', 'eta_s_est', 'tau_stim', 'w', 'tau_photon', 'S0gacterm', 'w_prime', 'Hsquared',
        'Homega', 'kappaL', 'Area', 'Imax', 'ImaxWPE', 'Ith', 'Pmax_est', 'Pmax_est_CW',
        'PmaxWPE_est', 'PmaxWPE_est_CW', 'detune_plot', 'gth_plot', 'del_gth', 'I_1', 'F_1',
        'tau_stim_1', 'S0gacterm_1', 'w_prime_1', 'Hsquared_1', 'Homega_1', 'I_2', 'F_2',
        'tau_stim_2', 'S0gacterm_2', 'w_prime_2', 'Hsquared_2', 'Homega_2', 'I_3', 'F_3',
        'tau_stim_3', 'S0gacterm_3', 'w_prime_3', 'Hsquared_3', 'Homega_3', 'I_4', 'F_4',
        'tau_stim_4', 'S0gacterm_4', 'w_prime_4', 'Hsquared_4', 'Homega_4', 'P_Test', 'I_5',
        'F_5', 'tau_stim_5', 'S0gacterm_5', 'w_prime_5', 'Hsquared_5', 'Homega_5'
    ]
    array_column_names = ['Guided_export', 'R_export', 'S_export', 'z', 'Guided_field']

    # Initialize the DataFrame with these columns
    scalar_results = pd.DataFrame(columns=scalar_column_names)
    
    # For results that are too complex for the dataframe. These can be used for plotting, but not optimization. 
    array_results = []
    
    # Performing row-wise calculations for each point
    num_rows = len(pickle_data['inputs'][next(iter(pickle_data['inputs']))])
    print(f"Number of rows is {num_rows}, i.e. unique simulation points.")
    for row in range(num_rows):
        # Extract row-specific data from hdf5_data
        row_parameters = {key: pickle_data['parameters'][key][row] for key in pickle_data['parameters']}
        row_inputs = {key: pickle_data['inputs'][key][row] for key in pickle_data['inputs']}
        row_results = {key: pickle_data['results'][key][row] for key in pickle_data['results']}

        # Initialize dictionaries to store results for this row
        scalar_row_results = {}
        array_row_results = {}

        # print(f"\nRow {row} data keys:")
        # print("Parameters keys:", list(row_parameters.keys()))
        # print("Inputs keys:", list(row_inputs.keys()))
        # print("Results keys:", list(row_results.keys()))
  
        # Unpacking finite-mode-solver results
        kappa_DFB = row_inputs['kappa_DFB']
        alpha_m = row_results['alpha_m']
        delta_m = row_results['delta_m']
        Gamma_lg = row_results['Gamma_lg']
        alpha_surf = row_results['alpha_surf']
        Guided_export = row_results['Guided_export']
        R_export = row_results['R_export']
        S_export = row_results['S_export']
        P_R = row_results['P_R']
        P_L = row_results['P_L']
        alpha_end = row_results['alpha_end']
        alpha_end_L = row_results['alpha_end_L']
        alpha_end_R = row_results['alpha_end_R']
        z = row_results['z']
                
        # If the mode solver failed, len(alpha_m) == 0. Set all variables to -1.
        if len(alpha_m) == 0:
            # Set scalar variables to -1
            scalar_row_results = {column_name: -1 for column_name in scalar_column_names}

            # Append the row to the DataFrame
            scalar_results = pd.concat([scalar_results, pd.DataFrame([scalar_row_results])], ignore_index=True)

            # Set array variables to arrays of -1 with the same shape as expected
            array_row_results = {}
            for column_name in array_column_names:
                if column_name in locals():
                    # Get the shape of the expected array
                    expected_shape = np.shape(locals()[column_name])
                    # Create an array of -1 with the same shape
                    array_row_results[column_name] = np.full(expected_shape, -1)
                else:
                    print(f"Warning: Array variable {column_name} not found in local scope")
                    # If we can't determine the shape, use a scalar -1
                    array_row_results[column_name] = -1

            array_results.append(array_row_results)
            
            print(f"Solution {row + 1} has len(alpha_m) == 0, i.e. the mode solver failed to find modes. All values set to -1; skipping.")
            continue
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #Calcs here
        
        alpha_i = alpha_w + alpha_bf
        alpha_opt = (2 * alpha_m) + alpha_i 
        k0 = 2 * np.pi / wavelength
        K0 = np.pi / Lambda
        energy = 1.24 / (wavelength * 10000)
        AReff = np.exp(-2*L*alpha_end_R)
        nu = c / wavelength
        dAR = Stage_t * NP 
        Lp = Stage_t 
        g_diffunit = g * 1e-5 ### converting units of differential gain
        gc = g_diffunit * 100 * q * Gamma / tau_up ### gain cross-section in cm/A
        sg = gc * Lp ### in cm^2 / A
        gth = (2 * alpha_m) + alpha_i
        I = P/(Wid * dAR * (1-AReff)) ### Circulating power at 1W
        F = I / (h * nu) ### Circulating photons / s in the cavity
        Jth = gth / (Gamma * g)
        lossterm = alpha_end_R / alpha_opt
        eta_s_est = energy * F1 * ni * lossterm * NP
        tau_stim = 1 / (F * sg ) ### stimulated lifetime
        w = np.pi * 2 * f ### Converting to angular frequency
        tau_photon = neff / (c * alpha_opt)
        S0gacterm = 1 / np.sqrt(tau_photon * tau_stim)
        w_prime = w / S0gacterm
        Hsquared = 1. / (w_prime**4 + (w_prime**2) * (tau_photon / tau_stim + 2. * tau_photon / tau_32 + (tau_photon * tau_stim) / (tau_32**2) - 2) + 1)
        Homega = np.sqrt(Hsquared)
        w_plot = np.arange(0, 2 * np.pi * 1e8, 2 * np.pi * 50e9)
        kappaL = kappa_DFB * L
            
        Area = L * Wid
        Imax = Jmax * Area * 1000
        ImaxWPE = JmaxWPE * Area * 1000
        Ith = Jth * Area * 1000
        
        Pmax_est = eta_s_est * (Imax - Ith)
        
        Pmax_est_CW  = Pmax_est * CW_Scaling_Factor
        PmaxWPE_est = ImaxWPE - Ith
        PmaxWPE_est_CW = PmaxWPE_est * CW_Scaling_Factor
        
        detune_plot = (2 * np.pi / (k0 + (delta_m / (wavelength/Lambda))) - wavelength) * 1e8
        gth_plot = gth
        
        # Assume the 0th mode is the fundamental
        fundamental_mode = 0
        Guided_field = Guided_export[fundamental_mode]
        
        
        del_gth = gth[1] - gth[0] #Intermodal gain between 1st and 0th mode

            
        ### For default pulsed power 
        I_1 = Pmax_est / (Wid * dAR * (1-AReff))
        F_1 = I_1 / (h * nu)
        tau_stim_1 = 1 / (F_1 * sg)
        S0gacterm_1 = 1 / np.sqrt(tau_photon * tau_stim_1)
        w_prime_1 = w / S0gacterm_1
        Hsquared_1 = 1. / (w_prime_1**4 + (w_prime_1**2) * (tau_photon / tau_stim_1 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_1) / (tau_32**2) - 2) + 1)
        Homega_1 = np.sqrt(Hsquared_1)
        
        ### For default pulsed power at peak WPE 
        I_2 = PmaxWPE_est / (Wid * dAR * (1-AReff))
        F_2 = I_2 / (h * nu)
        tau_stim_2 = 1 / (F_2 * sg)
        S0gacterm_2 = 1 / np.sqrt(tau_photon * tau_stim_2)
        w_prime_2 = w / S0gacterm_2
        Hsquared_2 = 1. / (w_prime_2**4 + (w_prime_2**2) * (tau_photon / tau_stim_2 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_2) / (tau_32**2) - 2) + 1)
        Homega_2 = np.sqrt(Hsquared_2)
        
        ### For CW-scaled power 
        I_3 = Pmax_est_CW / (Wid * dAR * (1-AReff))
        F_3 = I_3 / (h * nu)
        tau_stim_3 = 1 / (F_3 * sg)
        S0gacterm_3 = 1 / np.sqrt(tau_photon * tau_stim_3)
        w_prime_3 = w / S0gacterm_3
        Hsquared_3 = 1. / (w_prime_3**4 + (w_prime_3**2) * (tau_photon / tau_stim_3 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_3) / (tau_32**2) - 2) + 1)
        Homega_3 = np.sqrt(Hsquared_3)
        
        ### For CW-scaled power at peak WPE
        I_4 = PmaxWPE_est_CW / (Wid * dAR * (1-AReff))
        F_4 = I_4 / (h * nu)
        tau_stim_4 = 1 / (F_4 * sg)
        S0gacterm_4 = 1 / np.sqrt(tau_photon * tau_stim_4)
        w_prime_4 = w / S0gacterm_4
        Hsquared_4 = 1. / (w_prime_4**4 + (w_prime_4**2) * (tau_photon / tau_stim_4 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_4) / (tau_32**2) - 2) + 1)
        Homega_4 = np.sqrt(Hsquared_4)
        
        P_Test = 1.62 ### Testing a specific power  
        I_5 = P_Test / (Wid * dAR * (1-AReff))
        F_5 = I_5 / (h * nu)
        tau_stim_5 = 1 / (F_5 * sg)
        S0gacterm_5 = 1 / np.sqrt(tau_photon * tau_stim_5)
        w_prime_5 = w / S0gacterm_5
        Hsquared_5 = 1. / (w_prime_5**4 + (w_prime_5**2) * (tau_photon / tau_stim_5 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_5) / (tau_32**2) - 2) + 1)
        Homega_5 = np.sqrt(Hsquared_5)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # Unpacking values for saving. This is for values that were calculated for all N modes, but only require the fundamental solution.
        mode_index = 0
        alpha_m = alpha_m[mode_index]
        delta_m = delta_m[mode_index]
        Gamma_lg = Gamma_lg[mode_index]
        alpha_surf = alpha_surf[mode_index]
        Guided_export = Guided_export[mode_index, :] # Array
        R_export = R_export[mode_index, :] # Array
        S_export = S_export[mode_index, :] # Array
        P_R = P_R[mode_index] # Array
        P_L = P_L[mode_index] # Array 
        alpha_end = alpha_end[mode_index]
        alpha_end_L = alpha_end_L[mode_index]
        alpha_end_R = alpha_end_R[mode_index]   
        z = z # Array
        alpha_opt = alpha_opt[mode_index]
        AReff = AReff[mode_index]
        gth = gth[mode_index]
        I = I[mode_index]
        F = F[mode_index] 
        Jth = Jth[mode_index]
        lossterm = lossterm[mode_index] 
        eta_s_est = eta_s_est[mode_index]
        tau_stim = tau_stim[mode_index]
        tau_photon = tau_photon[mode_index]
        S0gacterm = S0gacterm[mode_index]
        w_prime = w_prime[mode_index]
        Hsquared = Hsquared[mode_index]
        Homega = Homega[mode_index]
        w_plot = w_plot[mode_index]
        Ith = Ith[mode_index]
        Pmax_est = Pmax_est[mode_index]
        Pmax_est_CW = Pmax_est_CW[mode_index]
        PmaxWPE_est = PmaxWPE_est[mode_index]
        PmaxWPE_est_CW = PmaxWPE_est_CW[mode_index]
        detune_plot = detune_plot[mode_index]
        gth_plot = gth_plot[mode_index]
                
        I_1 = I_1[mode_index]
        F_1 = F_1[mode_index]
        tau_stim_1 = tau_stim_1[mode_index]
        S0gacterm_1 = S0gacterm_1[mode_index]
        w_prime_1 = w_prime_1[mode_index]
        Hsquared_1 = Hsquared_1[mode_index]
        Homega_1 = Homega_1[mode_index]

        I_2 = I_2[mode_index]
        F_2 = F_2[mode_index]
        tau_stim_2 = tau_stim_2[mode_index]
        S0gacterm_2 = S0gacterm_2[mode_index]
        w_prime_2 = w_prime_2[mode_index]
        Hsquared_2 = Hsquared_2[mode_index]
        Homega_2 = Homega_2[mode_index]

        I_3 = I_3[mode_index]
        F_3 = F_3[mode_index]
        tau_stim_3 = tau_stim_3[mode_index]
        S0gacterm_3 = S0gacterm_3[mode_index]
        w_prime_3 = w_prime_3[mode_index]
        Hsquared_3 = Hsquared_3[mode_index]
        Homega_3 = Homega_3[mode_index]

        I_4 = I_4[mode_index]
        F_4 = F_4[mode_index]
        tau_stim_4 = tau_stim_4[mode_index]
        S0gacterm_4 = S0gacterm_4[mode_index]
        w_prime_4 = w_prime_4[mode_index]
        Hsquared_4 = Hsquared_4[mode_index]
        Homega_4 = Homega_4[mode_index]

        # P_Test is already a scalar, so no indexing needed

        I_5 = I_5[mode_index]
        F_5 = F_5[mode_index]
        tau_stim_5 = tau_stim_5[mode_index]
        S0gacterm_5 = S0gacterm_5[mode_index]
        w_prime_5 = w_prime_5[mode_index]
        Hsquared_5 = Hsquared_5[mode_index]
        Homega_5 = Homega_5[mode_index]
                            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        # I'll leave this in the script. If you're trying to identify what dataset to save some parameter into, it'll be printed out in this function; 
        # just uncomment the for loop. I'd recommend using this to check if you add any new calculations or variables and want to save them.
        # print("\nVariable types:")
        # for var_name, var_value in locals().items():
        #     # Skip JSONData, hdf5_data, inputs, and other non-relevant variables
        #     if var_name in ['JSONData', 'hdf5_data', 'inputs', 'row_parameters', 'row_inputs', 'row_results', 
        #                     'scalar_results', 'array_results', 'scalar_row_results', 
        #                     'array_row_results']:
        #         continue
            
        #     if isinstance(var_value, np.ndarray):
        #         print(f"{var_name}: Array of shape {var_value.shape}")
        #     elif isinstance(var_value, (list, tuple)):
        #         print(f"{var_name}: {type(var_value).__name__} of length {len(var_value)}")
        #     elif np.isscalar(var_value):
        #         print(f"{var_name}: Scalar ({type(var_value).__name__})")
        #     else:
        #         print(f"{var_name}: Other type ({type(var_value).__name__})")
                

        # Perform variable assignments to single_ and variable_ here
        scalar_row_results = {}
        for column_name in scalar_column_names:
            if column_name in locals():
                scalar_row_results[column_name] = locals()[column_name]
            elif column_name in globals():
                scalar_row_results[column_name] = globals()[column_name]
            else:
                print(f"Warning: Variable {column_name} not found in local or global scope")

        # Append the row to the DataFrame
        scalar_results = pd.concat([scalar_results, pd.DataFrame([scalar_row_results])], ignore_index=True)

        array_row_results = {}
        for column_name in array_column_names:
            if column_name in locals():
                array_row_results[column_name] = locals()[column_name]
            elif column_name in globals():
                array_row_results[column_name] = globals()[column_name]
            else:
                print(f"Warning: Array variable {column_name} not found in local or global scope")
            
        array_results.append(array_row_results)

        
    save_post_process(JSONData['results_fn'], scalar_results, array_results)

    return scalar_results, array_results

