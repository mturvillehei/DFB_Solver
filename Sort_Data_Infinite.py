import numpy as np
    
def safe_convert(value):
    try:
        return float(value)
    except ValueError:
        return complex(value.replace('i', 'j'))
    
def sort_Data_Infinite(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[5:]  # Skipping header lines
        data = []
        for line in lines:
            # Splitting each line by whitespace and converting to float or complex
            values = [safe_convert(value) for value in line.split()]
            data.append(values)
    return np.array(data, dtype=object)
    

def sort_nd(data, N, num_modes=None):
    # Determine unique values in the first N columns and their sizes
    params = [np.unique(data[:, n]) for n in range(N)]
    size_param = np.prod([len(p) for p in params])
    
    # Calculate number of modes if not provided
    if num_modes is None:
        num_modes = data.shape[0] // size_param
        if data.shape[0] % size_param != 0:
            print('The table may be incomplete.')
    
    # Create the sorted_data array with the appropriate shape
    sorted_data_shape = [len(p) for p in params] + [num_modes, data.shape[1]]
    sorted_data = np.zeros(sorted_data_shape, dtype=data.dtype)
    
    # Fill the sorted_data array
    for i in range(data.shape[0]):
        index = [np.where(params[n] == data[i, n])[0][0] for n in range(N)]
        mode_index = np.where(sorted_data[tuple(index) + (slice(None), N)] == 0)[0][0]
        sorted_data[tuple(index) + (mode_index,)] = data[i]
    
    return params, sorted_data

def process_infinite_data(params, sorted_data):
    
    print(sorted_data.shape)

    sorted_data_shape = sorted_data.shape

    DFB_mode_1 = np.zeros(sorted_data_shape[:-2] + (sorted_data_shape[-1],), dtype=np.complex128)
    DFB_mode_2 = np.zeros_like(DFB_mode_1, dtype=np.complex128)
    for i in range(sorted_data_shape[0]):
        for j in range(sorted_data_shape[1]):
            for k in range(sorted_data_shape[2]):
                # Extract the modes for the current indices
                modes = sorted_data[i, j, k, :, :]
                
                # Sort the modes based on the 14th column (Python index 13 since it's 0-based)
                sorted_modes = modes[modes[:, 13].argsort()[::-1]]  # This sorts in descending order; use [::1] for ascending
                
                # Perform the conditional assignment
                #### ~~~~~ If modes are being assigned incorrectly, flip this ~~~~~ ####
                if sorted_modes[0, 5] < sorted_modes[1, 5]:  # Python uses 0-based indexing, so column 6 is index 5
                    DFB_mode_2[i, j, k, :] = sorted_modes[0, :]
                    DFB_mode_1[i, j, k, :] = sorted_modes[1, :]
                else:
                    DFB_mode_2[i, j, k, :] = sorted_modes[1, :]
                    DFB_mode_1[i, j, k, :] = sorted_modes[0, :]
    return DFB_mode_1, DFB_mode_2

def Sort(filename, wavelength, Lambda):
    
    ND_n = 3 ### # of params, usually 3
    k0 = 2 * np.pi / wavelength
    K0 = np.pi / Lambda

    unsorted = sort_Data_Infinite(filename)
    params, sorted = sort_nd(unsorted, ND_n)  # ND_n is 3, as specified
    
    DFB_mode_1, DFB_mode_2 = process_infinite_data(params, sorted)
    
    alpha_DFB = np.zeros(DFB_mode_1.shape[:-1] + (2,), dtype=np.complex128)
    alpha_DFB[..., 0] = DFB_mode_2[..., 5]
    alpha_DFB[..., 1] = DFB_mode_1[..., 5]

    deltak_DFB = np.zeros_like(alpha_DFB)
    deltak_DFB[..., 0] = np.real(DFB_mode_2[..., 8]) / (K0 / 2 / k0)
    deltak_DFB[..., 1] = np.real(DFB_mode_1[..., 8]) / (K0 / 2 / k0)

    asurf_DFB = DFB_mode_2[..., 10]

    kappa_DFB = -(deltak_DFB[..., 0] - deltak_DFB[..., 1]) * K0 / 2 / k0 + 1j * (alpha_DFB[..., 0] - alpha_DFB[..., 1]) / 2
    zeta_DFB = -(deltak_DFB[..., 0] + deltak_DFB[..., 1]) * K0 / 2 / k0 + 1j * (alpha_DFB[..., 0] + alpha_DFB[..., 1]) / 2

    ### This needs to be updated once simulating actual DBR and SEDFB designs. '1.5' is arbitrary 
    ### and there will be detuning due to the change in DBR structure. 
    alpha_DBR = alpha_DFB + 1.5
    deltak_DBR = deltak_DFB
    asurf_DBR = asurf_DFB
    
    kappa_DBR = -(deltak_DBR[..., 0] - deltak_DBR[..., 1]) * K0 / 2 / k0 + 1j * (alpha_DBR[..., 0] - alpha_DBR[..., 1]) / 2
    zeta_DBR = -(deltak_DBR[..., 0] + deltak_DBR[..., 1]) * K0 / 2 / k0 + 1j * (alpha_DBR[..., 0] + alpha_DBR[..., 1]) / 2
    
    
    
    return {
        "params": params,
        "sorted": sorted,
        "alpha_DFB": alpha_DFB,
        "deltak_DFB": deltak_DFB,
        "asurf_DFB": asurf_DFB,
        "kappa_DFB": kappa_DFB,
        "zeta_DFB": zeta_DFB,
        "alpha_DBR": alpha_DBR,
        "deltak_DBR": deltak_DBR,
        "asurf_DBR": asurf_DBR,
        "kappa_DBR": kappa_DBR,
        "zeta_DBR": zeta_DBR,
        "DFB_mode_1": DFB_mode_1, 
        "DFB_mode_2": DFB_mode_2,
        "k0": k0,
        "K0": K0,
        "Lambda": Lambda
    }