import numpy as np
import matplotlib.pyplot as plt



def plot_mode_spectrum(detune_plot, gth, title='Mode Spectrum', xlabel='Detuning (Angstroms)', ylabel='g_{th} (cm^{-1})'):
    plt.figure()
    plt.plot(detune_plot, gth, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.gca().set_fontsize(14)
    plt.tight_layout()
def plot_field_intensity(z, Guided, L, title='Field Profile - Intensity R^2 + S^2', xlabel='Longitudinal position (mm)', ylabel='Field Intensity'):
    plt.figure()
    plt.plot(z, Guided, linewidth=1)
    plt.ylim([np.min(Guided) - np.std(Guided), np.max(Guided)+np.std(Guided)])
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
    """
    Creates a contour plot with the given data and labels.

    Parameters:
    - x: 1D array of x-axis values (e.g., cladding thickness).
    - y: 1D array of y-axis values (e.g., grating height).
    - z: 2D array of z-axis values to contour.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - cbar_label: Label for the colorbar.
    - xlim: Tuple specifying x-axis limits. Default is None.
    - levels: Number of levels for contouring. Default is 32.
    - colormap: Colormap to use for contouring. Default is 'parula'.
    """

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

def default_Calculations(Data):
    
    ### Loading parameters and Data from JSON output
    ### We have a full 2D plot for each design parameter combination, so these indices are chosen by 'results_indices'
    results_indices = Data['results_indices']
    plot_Results=Data['plot_Results'] ### Whether to plot the countour plots and mode spectrum
    plot_Indices=Data['plot_Indices'] ### Whether to plot the mesh showing indices (in order to find the right index for the 'results_indices' parameter.)
    L = 0.1 * Data['L'] ### cm used here, not mm
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
    tau_up = Data['tau_up'] ### tau_upper state
    tau_32 = Data['tau_32'] ### tau_3->2 lifetime
    neff = Data['neff']
    alpha_m = Data['alpha_m_all']
    CW_Scaling_Factor = Data['CW_Scaling_Factor']
    delta = Data['delta_m_all'][results_indices[0], results_indices[1], results_indices[2]] ### Detuning for the mode spectrum of the design at [results_indices]
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
    eta_s_est = np.where(valid_mask, F1 * ni * lossterm * NP, np.NaN)
    tau_stim = np.where(valid_mask, 1 / (F * sg ), np.NaN) ### stimulated lifetime
    w = np.pi * 2 * f ### Converting to angular frequency
    tau_photon = np.where(valid_mask, neff / (c * alpha_opt), np.NaN)
    S0gacterm = np.where(valid_mask, 1 / np.sqrt(tau_photon * tau_stim), np.NaN)
    w_prime = np.where(valid_mask, w / S0gacterm, np.NaN)
    Hsquared = np.where(valid_mask, 1. / (w_prime**4 + (w_prime**2) * (tau_photon / tau_stim + 2. * tau_photon / tau_32 + (tau_photon * tau_stim) / (tau_32**2) - 2) + 1), np.NaN)
    Homega = np.where(valid_mask, np.sqrt(Hsquared), np.NaN)
    w_plot = np.arange(0, 2 * np.pi * 1e8, 2 * np.pi * 50e9)
    
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
    gth_plot = gth[results_indices[0], results_indices[1], results_indices[2]]
    
    ### Careful with this! If the wrong mode is selected, the field profile plots will be for higher order junk modes
    mode_index = 0
    Guided_plot = Data['Guided_export_all'][results_indices[0], results_indices[1], results_indices[2], mode_index]
       
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



    ### Moving to plotting
    if plot_Indices:
        index_Plot(Data['cladding_thickness'], Data['grating_height'])
    
    params_indices = Data['params_indices']
    
    cladding_thickness = Data['cladding_thickness']
    cladding_thickness = cladding_thickness[params_indices[0][0]:params_indices[0][1]]
    grating_height = Data['grating_height']
    grating_height = grating_height[params_indices[1][0]:params_indices[1][1]]
    
    duty_cycle_indices = np.arange(params_indices[2][0], params_indices[2][1])
        
    if plot_Results:
        
        ### Comment this out for future sims, z should be stored in z_all
        z = np.linspace(0, 2*Data['L'], Data['num_Z'])
        

        ### NEED to fix field profiles and mode spectrum in Solve()
        #plot_mode_spectrum(detune_plot, gth_plot)
        
        #plot_field_intensity(z, Guided_plot, Data['L'])
        #plot_field_profile_real
        #plot_field_profile_abs
            
        for k in duty_cycle_indices:
           
            ### Contour Plots
            X, Y = np.meshgrid(cladding_thickness, grating_height)

            contour_plot(X, Y, tau_photon[:,:,k,0].T * 1e12, f"Photon Lifetime (ps) for duty cycle = {Data['duty_cycle'][k]}", 'Cladding Thickness (um)', 'Grating_Height (um)', 'Photon Lifetime (ps)')
            
            
        
        plt.show()
