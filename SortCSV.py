import numpy as np
import io
import pandas as pd
import os
from pickle_utils import save_sort


def safe_convert(value):
    try:
        return float(value)
    except ValueError:
        try:
            return complex(value.replace('i', 'j'))
        except ValueError:
            return value
        
def loadCSV(fn):
    with open(fn, 'r') as file:
            # Skip the first 4 lines
            for _ in range(4):
                next(file)
            lines = file.readlines()
            lines[0] = lines[0].lstrip('% ')
            cleaned_data = ''.join(lines)
            data_io = io.StringIO(cleaned_data)
            df = pd.read_csv(data_io)
            # Safely converting imaginary numbers
            df = df.applymap(safe_convert)
            return df

def process_group(group, headers=None):
    if len(group) < 2:
        print(f"Warning: Group has less than 2 rows: {group}")
        exit()

    mode1 = group.iloc[0]  
    mode2 = group.iloc[1]  
    
    if headers is None:
        
        # Mapping of headers - This will need to be re-written. Maybe have the column header mapping in the JSON file?
        gamma_col = 'dom2/dom3, Global Var Gamma' # Confinement factor Gamma
        delta_kappa_col = '(freq*2*pi/c_const-2*pi/wavelength[um])*wavelength/Period/2, delta kappa' # Delta kappa 
        loss_col = 'emw.damp/emw.freq/(Period[um]), Global Var Loss' # Global loss
        wavelength_col = 'c_const/emw.freq, Global Var wavelength'
        surf_loss_col = 'bnd1/dom1, Global Variable Surface Loss'
        sorting_col = gamma_col  # For identifying the first order DFB modes. Gamma and surface loss are traditionally used.

    else:
        gamma_col = headers['gamma_col']
        delta_kappa_col = headers['delta_kappa_col']
        loss_col = headers['loss_col']
        wavelength_col = headers['wavelength_col']
        surf_loss_col = headers['surf_loss_col']
        sorting_col = headers[headers['sorting_col_name']]

    # Sort by sorting_col. Pick the two with highest sorting_col value in this case.
    sorted_group = group.sort_values(sorting_col, ascending=False)
    
    mode1 = sorted_group.iloc[0]
    mode2 = sorted_group.iloc[1]

    try:
        
        alpha2_DFB = mode1[loss_col]
        alpha1_DFB = mode2[loss_col]
        
        deltak1_DFB = np.real(mode2[delta_kappa_col]) / (K0 / 2 / k0)
        deltak2_DFB = np.real(mode1[delta_kappa_col]) / (K0 / 2 / k0)
        
        asurf_DFB = mode2[surf_loss_col]
        
        
        
    except KeyError as e:
        print(f"Error {e} for modes with keys {mode1.columns}")
            
    kappa_DFB = -(deltak2_DFB - deltak1_DFB) * K0 / 2 / k0 + 1j * (alpha2_DFB - alpha1_DFB) / 2
    zeta_DFB = -(deltak2_DFB + deltak1_DFB) * K0 / 2 / k0 + 1j * (alpha2_DFB + alpha1_DFB) / 2
    alpha1_DBR = alpha1_DFB + 1.5
    alpha2_DBR = alpha2_DFB + 1.5
    deltak1_DBR = deltak1_DFB
    deltak2_DBR = deltak2_DFB
    asurf_DBR = asurf_DFB
    
    kappa_DBR = -(deltak2_DBR - deltak1_DBR) * K0 / 2 / k0 + 1j * (alpha2_DBR - alpha1_DBR) / 2
    zeta_DBR = -(deltak2_DBR + deltak1_DBR) * K0 / 2 / k0 + 1j * (alpha2_DBR + alpha1_DBR) / 2
    
    # Return a Series with the results
    return pd.Series({
        'alpha1_DFB': alpha1_DFB,
        'alpha2_DFB': alpha2_DFB,
        'deltak1_DFB': deltak1_DFB,
        'deltak2_DFB': deltak2_DFB,
        'asurf_DFB': asurf_DFB,
        'kappa_DFB': kappa_DFB,
        'zeta_DFB': zeta_DFB,
        'alpha1_DBR': alpha1_DBR,
        'alpha2_DBR': alpha2_DBR,
        'deltak1_DBR': deltak1_DBR,
        'deltak2_DBR': deltak2_DBR,
        'asurf_DBR': asurf_DBR,
        'kappa_DBR': kappa_DBR,
        'zeta_DBR': zeta_DBR,
    })


# Groups data by unique points in the parametric sweep.
# If adaptive mesh refinement was used, removes earlier iterations.
def sortND(df, output_fn, ND, ARM, headers=None):

    param_columns = list(df.columns[:ND]) 
    result_columns = list(df.columns[ND:])


    try:
        # Group by param_columns 
        df_grouped = df.groupby(param_columns)

        # Picks the last refinement level for each group
        if ARM:
            df_grouped = df_grouped.apply(lambda x: x[x['Refinement level'] == x['Refinement level'].max()])
            # df_grouped is now a DataFrame, not a GroupBy object
            results = []
            for name, group in df_grouped.groupby(param_columns):
                result = process_group(group, headers)
                results.append(pd.concat([pd.Series(name, index=param_columns), result]))
        else:
            # df_grouped is already grouped, so we can iterate over it directly
            results = []
            for name, group in df_grouped:
                result = process_group(group, headers)
                results.append(pd.concat([pd.Series(name, index=param_columns), result]))
        
        final_results = pd.DataFrame(results)
        #print(f"Final Results keys are {final_results.keys()}")
        
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        exit()

    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    # Saving
    save_sort(output_fn, param_columns, final_results)

                    
def SortCSV(filename, results_fn, wavelength, Lambda, ND, ARM):
    global k0
    global K0
    
    k0 = 2 * np.pi / wavelength
    K0 = np.pi / Lambda
    unsorted = loadCSV(filename)
    
    sortND(unsorted, results_fn, ND, ARM)

    