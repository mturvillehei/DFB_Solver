import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def find_results_Args(params):

    i_p = params['results_values'][0]
    j_p = params['results_values'][1]
    k_p = params['results_values'][2]
    
    ### Finding the index in ct, gh, dc that matches
    differences = np.abs(np.array(params['cladding_thickness']) - i_p)    
    i_c = np.argmin(differences) 
    ### If error is more than floating point
    if params['cladding_thickness'][i_c] - i_p > 0.001:
        raise Exception(f"for index {i_c} in cladding thickness, the best match is {params['cladding_thickness'][i_c]}, which does not meet the precision requirement.")
            
    differences = np.abs(np.array(params['grating_height']) - j_p)    
    j_c = np.argmin(differences)
    ### If error is more than floating point
    if params['grating_height'][j_c] - j_p > 0.001:
        raise Exception(f"for index {j_c} in grating_height, the best match is {params['grating_height'][j_c]}, which does not meet the precision requirement.")

    differences = np.abs(np.array(params['duty_cycle']) - k_p)    
    k_c = np.argmin(differences)        
    ### If error is more than floating point
    if params['duty_cycle'][k_c] - k_p > 0.001:
        raise Exception(f"for target {k_p} in duty cycle, the best match is {params['duty_cycle'][k_c]}, which does not meet the precision requirement.")

    ### Finding the index in the results (via params_indices) that matches 
    ### Since the original indices are stored, I can just use the same method to find the correct index (i.e. if index 17 is found, search for the results 
    ### index that = 17)
    i_r = np.argmin(params['params_indices'][0] - i_c)
        ### This won't happen if the results_value is in the sweep results
    if params['params_indices'][0][i_r] - i_c > 0.5:
        raise Exception(f"For cladding thickness results target {i_p}, with target index {i_c}, closest match is {i_r}, and not in results.")
    
    j_r = np.argmin(params['params_indices'][1] - j_c)
        ### This won't happen if the results_value is in the sweep results
    if params['params_indices'][1][j_r] - j_c > 0.5:
        raise Exception(f"For grating height results target {j_p}, with target index {j_c}, closest match is {j_r}, and not in results.")
    
    k_r = np.argmin(params['params_indices'][2] - k_c)
        ### This won't happen if the results_value is in the sweep results
    if params['params_indices'][2][k_r] - k_c > 0.5:
        raise Exception(f"For duty cycle results target {k_p}, with target index {k_c}, closest match is {k_r}, and not in results.")
    
    
    return i_r, j_r, k_r, i_c, j_c, k_c

def plot_mode_spectrum(detune_plot, gth, title='Mode Spectrum', xlabel='Detuning (Angstroms)', ylabel='g_{th} (cm^{-1})'):
    plt.figure()
    plt.plot(detune_plot, gth, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    ### For identifying the mode index when index 0 isn't the fundamental lateral mode
    # for i, (x, y) in enumerate(zip(detune_plot, gth)):
    #     plt.annotate(str(f"Mode index {i}"), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
def plot_field_intensity(z, Guided, L, title, xlabel='Longitudinal position (mm)', ylabel='Field Intensity'):
    plt.figure()
    plt.plot(z, Guided, linewidth=1)
    plt.ylim([0, 2*np.max(Guided)])
    plt.xlim([0, L])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(['Intensity R^2 + S^2'])
    #plt.gca().set_fontsize(14)
    plt.tight_layout()
def plot_field_profile_real(z, R, S, L, title='Field Profile (Real part)', xlabel='Longitudinal position (mm)'):
    plt.figure()
    plt.plot(z, R, z, S, linewidth=1)
    plt.xlim([0, L])
    plt.xlabel(xlabel)
    plt.legend(['Right-moving wave R(z)', 'Left-moving wave S(z)'], loc='best')
    plt.title(title)
    #plt.gca().set_fontsize(14)
    plt.tight_layout()
def plot_field_profile_abs(z, R, S, L, title='Field Profile', xlabel='Longitudinal position (mm)'):
    plt.figure()
    plt.plot(z, np.abs(R), z, np.abs(S), linewidth=1)
    plt.xlim([-L/2*10, L/2*10])
    plt.xlabel(xlabel)
    plt.legend(['Right-moving wave R(z)', 'Left-moving wave S(z)'], loc='best')
    plt.title(title)
    #plt.gca().set_fontsize(14)
    plt.tight_layout()

    
def contour_plot(x, y, z, title, xlabel, ylabel, cbar_label, xlim=None, levels=32, colormap='plasma'):
    
    ### https://stackoverflow.com/questions/34859628/has-someone-made-the-parula-colormap-in-matplotlib/43264077#43264077
    cm_data = [[0.2422, 0.1504, 0.6603],
    [0.2444, 0.1534, 0.6728],
    [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],
    [0.2522, 0.1689, 0.7179],
    [0.254, 0.1732, 0.7286],
    [0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.761],
    [0.2611, 0.1893, 0.7719],
    [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],
    [0.2661, 0.2011, 0.8043],
    [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],
    [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528],
    [0.274, 0.228, 0.8612],
    [0.2749, 0.233, 0.8692],
    [0.2758, 0.2382, 0.8767],
    [0.2766, 0.2435, 0.884],
    [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],
    [0.2788, 0.2598, 0.9035],
    [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.915],
    [0.2802, 0.2764, 0.9204],
    [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305],
    [0.2811, 0.293, 0.9352],
    [0.2813, 0.2985, 0.9397],
    [0.2814, 0.304, 0.9441],
    [0.2814, 0.3095, 0.9483],
    [0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563],
    [0.2809, 0.3259, 0.96],
    [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],
    [0.2189, 0.462, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],
    [0.1785, 0.5296, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.154, 0.5902, 0.9218],
    [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905],
    [0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.722, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.751, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.517],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],
    [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717],
    [0.771, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],
    [0.806, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695],
    [0.9, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.828, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.884, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.148],
    [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.973, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805]]
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    colormap = parula_map
    
    fig, ax = plt.subplots()
    contour_set = ax.contourf(x, y, z, levels, cmap=colormap, linestyle='none')
    cbar = fig.colorbar(contour_set)
    cbar.set_label(cbar_label)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    
    
def spectrum_plot():
    return

def index_Plot(cladding_thickness, grating_height):
    # Generating meshgrid for plotting
    X, Y = np.meshgrid(cladding_thickness, grating_height)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'ko', linestyle="dashed", markersize=5, label='Parameter Points')
    for i, x_val in enumerate(cladding_thickness):
        for j, y_val in enumerate(grating_height):
            plt.text(x_val, y_val, f"({i},{j})", fontsize=9, ha='right')

    plt.xlabel('Cladding Thickness')
    plt.ylabel('Grating Height')
    plt.title('Parameter Sweep Indices')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    return

def default_Calculations(Data, sweep):
    
    ### Loading parameters and Data from JSON output
    
    ### We have a full 2D plot for each design parameter combination, so these indices are chosen by 'results_indices'
    params_indices = Data['params_indices']
    results_indices = find_results_Args(Data)
    plot_Results=Data['plot_Results'] ### Whether to plot the countour plots and mode spectrum
    
    if sweep:
        plot_Results = False
    
    L = Data['L'] ### cm used here, not mm
    f = Data['f'] ### frequency for h(w) calculation
    Gamma = Data['Gamma']
    alpha_w = Data['alpha_w'] ### Waveguide loss. Calculated from 2D cross-section simulation.
    alpha_bf = Data['alpha_bf'] ### backfilling loss,  assumed to be 0.1 by Dan
    Jmax = Data['Jmax'] ### Peak current 
    JmaxWPE = Data['JmaxWPE'] ### Jmax at peak WPE 
    g = Data['g'] ### Differential gain. See IFR_Differential_Gain (email me if you need it)
    wavelength = Data['wavelength'] ### wavelength 
    Lambda = Data['Lambda'] ### grating length
    ni = Data['ni'] ### internal efficiency
    NP = Data['NP'] ### number of stages
    F1 = Data['F1'] ### used for calculating slope efficiency. not sure where this is from 
    HR = Data['HR'] ### HR percent on the back facet
    alpha_end_R = Data['alpha_end_R_all'] ### mirror loss, front facet
    alpha_end_L = Data['alpha_end_L_all'] ### mirror loss, back facet
    q = Data['q'] ### charge unit
    h = Data['h'] ### planck's
    c = Data['c'] ### speed of light
    P = Data['P'] ### power (for calculating circulating power)
    Wid = Data['Wid'] ### device width
    Stage_t = Data['stage_t'] ### stage thickness
    tau_up = Data['tau_up'] ### tau_upper state -> tau,ul,global [IFR + AD + LO]
    #tau_32 = Data['tau_32'] ### tau_3->2 lifetime
    tau_32 = tau_up
    neff = Data['neff']
    alpha_m = Data['alpha_m_all']
    CW_Scaling_Factor = Data['CW_Scaling_Factor']
    kappa = np.array(Data['kappa_DFB'])
    ### Let's say we only care about the central 5 indices in a sweep with 100 different points - this allows us to choose only the central indices
    ### Same structure as in EEDFB_Finite_Mode_Solver
    
    #results_indices = [results_indices[0] - params_indices[0][0], results_indices[1] - params_indices[1][0], results_indices[2] - params_indices[2][0]]
    
    delta = Data['delta_m_all'][results_indices[3], results_indices[4], results_indices[5]] ### Detuning for the mode spectrum of the design at [results_indices]
    valid_mask = ~np.isnan(alpha_m) ### Only performing calculations on indices where a mode was found. np.NaN will raise an error if we try to run element-wise 
    
    ### Default calculations
    alpha_i = alpha_w + alpha_bf ### Internal loss.
    alpha_opt = np.where(valid_mask, (2 * alpha_m) + alpha_i, np.NaN) ### optical loss
    k0 = 2 * np.pi / wavelength ### prop constant k of the omode
    K0 = np.pi / Lambda ### prop constant k of the grating
    energy = 1.24 / (wavelength * 10000)
    AReff = np.where(valid_mask, np.exp(-2*L*alpha_end_R), np.NaN) ### Effective mirror reflectivity of the front facet
    nu = c / wavelength ### frequency
    dAR = Stage_t * NP ### Differential thickness
    Lp = Stage_t 
    g_diffunit = g * 1e-5 ### converting units of differential gain
    gc = g_diffunit * 100 * q * Gamma / tau_up ### gain cross-section in cm/A
    sg = gc * Lp ### in cm^2 / A
    gth = np.where(valid_mask, (2 * alpha_m) + alpha_i, np.NaN)
    I = np.where(valid_mask, P/(Wid * dAR * (1-AReff)), np.NaN) ### Circulating power at 1W
    F = np.where(valid_mask, I / (h * nu), np.NaN) ### Circulating photons / s in the cavity
    Jth = np.where(valid_mask, gth / (Gamma * g), np.NaN)
    lossterm = np.where(valid_mask, alpha_end_R / alpha_opt, np.NaN)
    eta_s_est = np.where(valid_mask, energy * F1 * ni * lossterm * NP, np.NaN)
    tau_stim = np.where(valid_mask, 1 / (F * sg ), np.NaN) ### stimulated lifetime
    w = np.pi * 2 * f ### Converting to angular frequency
    tau_photon = np.where(valid_mask, neff / (c * alpha_opt), np.NaN)
    S0gacterm = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim), np.NaN)
    w_prime = np.where(valid_mask, w / S0gacterm, np.NaN)
    Hsquared = np.where(valid_mask, 1. / (w_prime**4 + (w_prime**2) * (tau_photon / tau_stim + 2. * tau_photon / tau_32 + (tau_photon * tau_stim) / (tau_32**2) - 2) + 1), np.NaN)
    Homega = np.where(valid_mask, np.sqrt(Hsquared), np.NaN)
    w_plot = np.arange(0, 2 * np.pi * 1e8, 2 * np.pi * 50e9)
    kappaL = np.where(valid_mask[:,:,:, 0], kappa * L, np.NaN)
    # S0gacterm_plot = S0gacterm[:,:,:,1] ### Might have a dimension mismatch here?
    # S0gacterm_plot_expanded = S0gacterm_plot[:,:,:,np.newaxis]
    # wprime_plot = np.where(valid_mask[:,:,:,np.newaxis], w_plot / S0gacterm_plot_expanded, np.NaN)
    
    Area = L * Wid
    Imax = Jmax * Area * 1000
    ImaxWPE = JmaxWPE * Area * 1000
    Ith = np.where(valid_mask, Jth * Area * 1000, np.NaN)
    
    Pmax_est = np.where(valid_mask, eta_s_est * (Imax - Ith), np.NaN)
    
    Pmax_est_CW  = np.where(valid_mask, Pmax_est * CW_Scaling_Factor, np.NaN) 
    PmaxWPE_est = np.where(valid_mask, ImaxWPE - Ith, np.NaN)
    PmaxWPE_est_CW = np.where(valid_mask, PmaxWPE_est * CW_Scaling_Factor, np.NaN)
       
    detune_plot = (2 * np.pi / (k0 + (delta / (wavelength/Lambda))) - wavelength) * 1e8
    gth_plot = gth[results_indices[3], results_indices[4], results_indices[5]]
    
    ### It would probably be best to compare results and check which mode we want during post-processing -> GUI prompt to select mode index 
    ### based on field profile.
    mode_index = 0
    Guided_plot = Data['Guided_export_all'][results_indices[3], results_indices[4], results_indices[5], mode_index]

    # for i in range(len(Data['cladding_thickness'])):
    #     for j in range(len(Data['grating_height'])):
    #         for k in range(len(Data['duty_cycle'])):
    #             #print(gth[i])

    del_gth = np.zeros((gth.shape[0], gth.shape[1], gth.shape[2]))
    del_gth = gth[:,:,:,1] - gth[:,:,:,0]

    ### For default pulsed power 
    I_1 = np.where(valid_mask, Pmax_est / (Wid * dAR * (1-AReff)), np.NaN)
    F_1 = np.where(valid_mask, I_1 / (h * nu), np.NaN)
    tau_stim_1 = np.where(valid_mask, 1 / (F_1 * sg), np.NaN)
    S0gacterm_1 = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim_1), np.NaN)
    w_prime_1 = np.where(valid_mask, w / S0gacterm_1, np.NaN)
    Hsquared_1 = np.where(valid_mask, 1. / (w_prime_1**4 + (w_prime_1**2) * (tau_photon / tau_stim_1 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_1) / (tau_32**2) - 2) + 1), np.NaN)
    Homega_1 = np.where(valid_mask, np.sqrt(Hsquared_1), np.NaN)
    
    ### For default pulsed power at peak WPE 
    I_2 = np.where(valid_mask, PmaxWPE_est / (Wid * dAR * (1-AReff)), np.NaN)
    F_2 = np.where(valid_mask, I_2 / (h * nu), np.NaN)
    tau_stim_2 = np.where(valid_mask, 1 / (F_2 * sg), np.NaN)
    S0gacterm_2 = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim_2), np.NaN)
    w_prime_2 = np.where(valid_mask, w / S0gacterm_2, np.NaN)
    Hsquared_2 = np.where(valid_mask, 1. / (w_prime_2**4 + (w_prime_2**2) * (tau_photon / tau_stim_2 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_2) / (tau_32**2) - 2) + 1), np.NaN)
    Homega_2 = np.where(valid_mask, np.sqrt(Hsquared_2), np.NaN)
    
    ### For CW-scaled power 
    I_3 = np.where(valid_mask, Pmax_est_CW / (Wid * dAR * (1-AReff)), np.NaN)
    F_3 = np.where(valid_mask, I_3 / (h * nu), np.NaN)
    tau_stim_3 = np.where(valid_mask, 1 / (F_3 * sg), np.NaN)
    S0gacterm_3 = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim_3), np.NaN)
    w_prime_3 = np.where(valid_mask, w / S0gacterm_3, np.NaN)
    Hsquared_3 = np.where(valid_mask, 1. / (w_prime_3**4 + (w_prime_3**2) * (tau_photon / tau_stim_3 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_3) / (tau_32**2) - 2) + 1), np.NaN)
    Homega_3 = np.where(valid_mask, np.sqrt(Hsquared_3), np.NaN)
    
    ### For CW-scaled power at peak WPE
    I_4 = np.where(valid_mask, PmaxWPE_est_CW / (Wid * dAR * (1-AReff)), np.NaN)
    F_4 = np.where(valid_mask, I_4 / (h * nu), np.NaN)
    tau_stim_4 = np.where(valid_mask, 1 / (F_4 * sg), np.NaN)
    S0gacterm_4 = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim_4), np.NaN)
    w_prime_4 = np.where(valid_mask, w / S0gacterm_4, np.NaN)
    Hsquared_4 = np.where(valid_mask, 1. / (w_prime_4**4 + (w_prime_4**2) * (tau_photon / tau_stim_4 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_4) / (tau_32**2) - 2) + 1), np.NaN)
    Homega_4 = np.where(valid_mask, np.sqrt(Hsquared_4), np.NaN)
      
    P_Test = 1.62 ### Testing a specific power  
    I_5 = np.where(valid_mask, P_Test / (Wid * dAR * (1-AReff)), np.NaN)
    F_5 = np.where(valid_mask, I_5 / (h * nu), np.NaN)
    tau_stim_5 = np.where(valid_mask, 1 / (F_5 * sg), np.NaN)
    S0gacterm_5 = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim_5), np.NaN)
    w_prime_5 = np.where(valid_mask, w / S0gacterm_5, np.NaN)
    Hsquared_5 = np.where(valid_mask, 1. / (w_prime_5**4 + (w_prime_5**2) * (tau_photon / tau_stim_5 + 2. * tau_photon / tau_32 + (tau_photon * tau_stim_5) / (tau_32**2) - 2) + 1), np.NaN)
    Homega_5 = np.where(valid_mask, np.sqrt(Hsquared_5), np.NaN)


    params_indices = Data['params_indices']
    
    cladding_thickness = np.array(Data['cladding_thickness'])
    cladding_thickness = cladding_thickness[params_indices[0]]
    grating_height = np.array(Data['grating_height'])
    grating_height = grating_height[params_indices[1]]
    duty_cycle = results_indices[5]
        
    if plot_Results:
        
        ### Comment this out for future sims, z should be stored in z_all
        z = np.linspace(0, 2*Data['L'], Data['num_Z'])
        plot_mode_spectrum(detune_plot, gth_plot)

        plot_field_intensity(z, Guided_plot, Data['L'], f'Field Profile - Intensity R^2 + S^2')

        ### Contour Plots
        X, Y = np.meshgrid(cladding_thickness, grating_height)
        contour_plot(X, Y, tau_photon[:,:,duty_cycle,mode_index].T * 1e12, f"Photon Lifetime for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Photon Lifetime (ps)')
        contour_plot(X, Y, eta_s_est[:,:,duty_cycle,mode_index].T, f"Estimated slope efficiency for for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Slope Efficiency (W/A)')
        contour_plot(X, Y, del_gth[:,:,duty_cycle].T, f"Intermodal discrimination for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Intermodal discrimination cm^-1')
        contour_plot(X, Y, Jth[:,:,duty_cycle,mode_index].T, f"Threshold current density for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Threshold current density kA/cm^2')
        contour_plot(X, Y, Pmax_est[:,:,duty_cycle,mode_index].T, f"Pmax for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Pmax (W)')
        contour_plot(X, Y, AReff[:,:,duty_cycle,mode_index].T , f"AReff for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'AReff')
        contour_plot(X, Y, alpha_m[:,:,duty_cycle,mode_index].T , f"Fundamental mode loss for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Alpha')
        contour_plot(X, Y, S0gacterm[:,:,duty_cycle,mode_index].T /(2*np.pi), f"S0Gac term for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'S0Gac term (Hz)')
        contour_plot(X, Y, Homega[:,:,duty_cycle,mode_index].T, f"Homega for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Homega')
        contour_plot(X, Y, Homega_1[:,:,duty_cycle,mode_index].T, f"Homega Pulsed for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Homega Pulsed')
        contour_plot(X, Y, Homega_2[:,:,duty_cycle,mode_index].T, f"Homega Pulsed+WPE for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Homega Pulsed+WPE')
        contour_plot(X, Y, Homega_3[:,:,duty_cycle,mode_index].T, f"Homega CW for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Homega CW')
        contour_plot(X, Y, Homega_4[:,:,duty_cycle,mode_index].T, f"Homega CW+WPE for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Homega CW+WPE')
        contour_plot(X, Y, alpha_m[:,:,duty_cycle,mode_index].T, f"Modal loss for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Modal loss cm^-1')
        contour_plot(X, Y, kappaL[:,:,duty_cycle].T, f"Kappa * L for duty cycle = {Data['duty_cycle'][results_indices[5]]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'kappa * L')
        
        plt.show()

# headers = ['rR', 'duty_cycle', 'cladding_thickness', 
#            'grating_height', 'PmaxWPE_est_CW', 'kappa_DFB_L', 'S0gacterm', 
#            'AReff', 'Homega_4', 'alpha_m', 'Jth', 'del_gth', 'eta_s_est', 
#            'tau_photon', 'tau_stim_4', 'Guided_plot_abs'
# ]
    return (Data['rR'], 
            Data['duty_cycle'][results_indices[5]], 
            Data['cladding_thickness'][results_indices[3]], 
            Data['grating_height'][results_indices[4]], 
            PmaxWPE_est_CW[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            np.abs(Data['kappa_DFB'][results_indices[3]][results_indices[4]][results_indices[5]]) * Data['L'],
            S0gacterm[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            AReff[results_indices[3], results_indices[4], results_indices[5], mode_index],
            Homega_4[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            alpha_m[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            Jth[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            del_gth[results_indices[3], results_indices[4], results_indices[5]], 
            eta_s_est[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            tau_photon[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            tau_stim_4[results_indices[3], results_indices[4], results_indices[5], mode_index], 
            np.abs(Guided_plot[-1]))