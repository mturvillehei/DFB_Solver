import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sc
from scipy.optimize import fsolve
import numpy as np
import warnings

def gamma_DFB(kappa, alpha, delta):
    return np.sqrt((kappa**2) + (alpha + 1j*delta)**2 ) 

def alpha_prime(alpha, zeta):
    return alpha - np.imag(zeta)

def delta_prime(delta, zeta):
    return delta + np.real(zeta)

#### F_DFB_matrix Input = [kappa, gamma, L]
def F_DFB_matrix(alpha, delta, kappa, gamma, L):
    F = np.array([
        [np.cosh(gamma*L) + (alpha + 1j*delta)/gamma*np.sinh(gamma*L), 1j*kappa/gamma*np.sinh(gamma*L)],
        [-1j*kappa/gamma*np.sinh(gamma*L), np.cosh(gamma*L) - (alpha + 1j*delta)/gamma*np.sinh(gamma*L)]
    ])
    return F
    
def Fr(r):
    inv_denom = 1 / (1 - r)
    return np.array([
        [inv_denom, -r * inv_denom],
        [-r * inv_denom, inv_denom]
    ])
    
def Fp(phi):
    return [np.exp(1j*phi),
            0,
            0,
            np.exp(-1j*phi)      
    ]


### 1 S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003)
def F_DFB(alpha, delta, kappa0, zeta0, L0, rR, rL, asurf, Lambda):
    #inputs = (kappa_DFB, zeta_DFB, L1, rR, rL, asurf, Lambda)
    alpha_prime_DFB = alpha_prime(alpha, zeta0) ### Converting to alpha prime, delta prime
    delta_prime_DFB = delta_prime(delta, zeta0)
    gamma = gamma_DFB(kappa0, alpha_prime_DFB, delta_prime_DFB)
          
    #### F_DFB_matrix Input = [kappa, gamma, L]

    F = Fr(rR) @ F_DFB_matrix(alpha_prime_DFB, delta_prime_DFB, kappa0, gamma, L0) @ F_DFB_matrix(alpha_prime_DFB, delta_prime_DFB, kappa0, gamma, L0) @ np.linalg.inv(Fr(rL))
    return F


def F22(F):
    return F[1, 1] 
    
def Fvec_solv(x, *args):
    
    alpha, delta = x
    F = F_DFB(alpha, delta, *args)
    F22_value = F22(F)
    
    return [np.real(F22_value), np.imag(F22_value)]

### Right and Left moving wave solutions for the coupled wave approach
def Fdfb_R(z, kappa, alpha, delta, gamma, R, S):
    return (np.cosh(gamma*z) + np.sinh(gamma*z)*(alpha + 1j * delta) / gamma)*R + (1j*kappa*np.sinh(gamma*z)/gamma)*S

def Fdfb_S(z, kappa, alpha, delta, gamma, R, S):
    return (np.cosh(gamma*z) - np.sinh(gamma*z)*(alpha + 1j * delta) / gamma)*S + (-1j*kappa*np.sinh(gamma*z)/gamma)*R
    
    
#### In the Matlab script, this is called 'Find Modes', and is called to execute the transfer matrix method.
#### Following execution, results are passed into the coupled wave solver.   
def Solver_Loop(inputs, num_Modes):
    
    alpha0 = []
    delta = []
    guess_all = []
    norm_guess_all = []
 
    #inputs = (kappa_DFB, zeta_DFB, L1, rR, rL)
    kappa0 = inputs[0]
    zeta0 = inputs[1]
    L = inputs[2]
    rR = inputs[3]
    rL = inputs[4]
    
    for j in np.arange(0, np.ceil(np.abs(np.imag(kappa0)) * L)* 2 +5, 0.3):
        
        for i in np.arange(-max(np.ceil(np.abs(np.real(kappa0)) * L) + 20, 20), 
                            max(np.ceil(np.abs(np.real(kappa0)) * L) + 10, 20), 0.3):
            
            ### Identifying guess from kappa, zeta (iterating through possible values up to 2imag_kappa + 5, real_kapp + 10
            ### Converting to mm^-1 (Should I use cm here?)
            guess = [j / L, i / L - np.real(zeta0)]
            guess_all.append(guess)
            norm_guess = np.linalg.norm(Fvec_solv(guess, *inputs))
            norm_guess_all.append(norm_guess)
            #print(norm_guess)

            min_Fval = max(300, np.abs(kappa0)**2 * 10)
            if not alpha0 or guess[0] < np.min(alpha0):
                min_Fval *= 100
            #print(min_Fval)
            
            ### If the norm guess of Fvec_solv is less than the max of 300, np.abs(kappa0)**2 * 10 (300 is arbitrary?) continue
            if norm_guess < min_Fval:
                solution, info, ier, mesg = fsolve(Fvec_solv, guess, args=inputs, full_output=True, xtol=1E-20, epsfcn=1E-20)
                #print(solution)
                
                # if ier != or np.linalg.norm(info['fvec']) > 1e-5:
                #     continue
                
                ### If alpha0 is empty, or if the functions are not close enoguh to the target
                ### "In summary, the expression is used to determine if the current solution found by fsolve is essentially a duplicate of any solution found in previous iterations, based on a specified tolerance (0.01). If it returns True, the current solution is very close to at least one previously found solution."
                # print(f"alpha0 {alpha0}")
                # print(f"Vert stack transposed: {np.vstack([alpha0, delta]).T}")
                # print(solution)
                if not alpha0 or not np.any(np.all(np.abs(np.vstack([alpha0, delta]).T - solution) < 0.01, axis=1)):
                    alpha0.append(solution[0])
                    delta.append(solution[1])
                
                ### If not empty
                elif alpha0:
                    index = int(np.where(np.prod(np.abs(np.vstack([alpha0, delta]).T - solution) < 0.01, axis=1))[0])
                    ### If solution is satisfactory, i.e. better than previous solution, 
                    #print(f"solution: {np.abs(Fvec_solv(solution, *inputs))}")
                    #print(f"alpha0 delta index solutions: {np.abs(Fvec_solv([alpha0[index], delta[index]], *inputs))}")
                    if np.abs(Fvec_solv(solution, *inputs))[0] < np.abs(Fvec_solv([alpha0[index], delta[index]], *inputs))[0]:
                        ### Then store the new results
                        alpha0[index] = solution[0]
                        delta[index] = solution[1]
        ### j = num modes found for this iteration
        if len(alpha0) > 2 and np.max(alpha0) < (j - 20) / L:
            #print(j)
            break
            
    if alpha0:
        ### Sorting results by loss, and then resorting
        sorted_indices = np.argsort(alpha0)
        alpha0 = np.array(alpha0)[sorted_indices]
        delta = np.array(delta)[sorted_indices]
        num_modes = ~np.isnan(alpha0)
        alpha0 = alpha0[num_modes]
        delta = delta[num_modes]       
    
    #print(f"alpha0 is {alpha0} with {num_modes} found")
    #print(f"delta is {delta} with {num_modes} found")    
    return alpha0, delta

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Roughly lines 250 and on, in the finite_solver_trans_3d_sweep_DFB_test_mod.m file
### To integrate DBR regions, add the lines for the DBR, trans, and phase shift regions [Rd1, Sd2, ... etc]. 
def Sg(z, L1, kappa, alpha, delta, gamma, Rd, Sd):
    if z <= L1:
        return S1(z, kappa, alpha, delta, gamma, Rd, Sd)
    else:
        return S2(z, kappa, alpha, delta, gamma, Rd, Sd)
def Rg(z, L1, kappa, alpha, delta, gamma, Rd, Sd):
    if z <= L1:
        return R1(z, kappa, alpha, delta, gamma, Rd, Sd)
    else:
        return R2(z, kappa, alpha, delta, gamma, Rd, Sd)
def Sn(z, L1, kappa, alpha, delta, gamma, Rd, Sd, asurf_1):
    if z <= L1:
        return S1(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
    else:
        return S2(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
def Rn(z, L1, kappa, alpha, delta, gamma, Rd, Sd, asurf_1):
    if z <= L1:
        return R1(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
    else:
        return R2(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1

def S1(z, kappa, alpha, delta, gamma, Rd, Sd):
    return Fdfb_S(z, kappa, alpha, delta, gamma, Rd, Sd)
def S2(z, kappa, alpha, delta, gamma, Rd, Sd):
    return Fdfb_S(z, kappa, alpha, delta, gamma, Rd, Sd)
def R1(z, kappa, alpha, delta, gamma, Rd, Sd):
    return Fdfb_R(z, kappa, alpha, delta, gamma, Rd, Sd)
def R2(z, kappa, alpha, delta, gamma, Rd, Sd):
    return Fdfb_R(z, kappa, alpha, delta, gamma, Rd, Sd)

def Sg1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd):
    if z <= L1:
        return S1(z, kappa, alpha, delta, gamma, Rd, Sd)
    else:
        return S2(z, kappa, alpha, delta, gamma, Rd, Sd)
def Rg1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd):
    if z <= L1:
        return R1(z, kappa, alpha, delta, gamma, Rd, Sd)
    else:
        return R2(z, kappa, alpha, delta, gamma, Rd, Sd)
def Sn1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf_1):
    if z <= L1:
        return S1(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
    else:
        return S2(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
def Rn1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf_1):
    if z <= L1:
        return R1(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
    else:
        return R2(z, kappa, alpha, delta, gamma, Rd, Sd) * asurf_1
    
def Guided(z, S1, S2, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf):
    return np.abs(Rg1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd))**2 + np.abs(Sg1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd))**2
def Near(z, S1, S2, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf):
    return np.abs(Sg1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd) + Rg1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd))**2
def Near2(z, S1, S2, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf):
    return np.abs(Sn1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf) + Rn1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf))**2
def NearAmp(z, S1, S2, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf):
    return Sn1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf) + Rn1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf)
def Antisym(z, S1, S2, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd, asurf):
    return np.abs(Sg1(z, S1, S2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd) - Rg1(z, R1, R2, L1, Lambda, kappa, alpha, delta, gamma, Rd, Sd))**2


def Coupled_Wave_Solver(alpha_m, delta_m, inputs, num):
    with warnings.catch_warnings():
        ### Ignoring the obnoxious warning about 'Casting complex values to reals'
        warnings.simplefilter("ignore", category = np.ComplexWarning)
        
        #num = 500 ### Resolution of z
        #inputs = (kappa_DFB, zeta_DFB, L1, rR, rL, asurf, Lambda)
        kappa0 = inputs[0]
        zeta1 = inputs[1]
        L1 = inputs[2]
        rR = inputs[3]
        rL = inputs[4]
        asurf_DFB = inputs[5]
        Lambda = inputs[6]
        
        alpha_surf = []
        efficiency = []
        Ratio_all = []
        DBR_ratio_ = []
        peak_ratio = []
        peak_angle = []
        
        Guided_export = np.zeros((len(alpha_m), num))
        Z_export = np.zeros((len(alpha_m), num))
        R_export = np.zeros((len(alpha_m), num))
        S_export = np.zeros((len(alpha_m), num))
        Gamma_lg = np.zeros(len(alpha_m))
        alpha_end = np.zeros(len(alpha_m))
        P_R = np.zeros(len(alpha_m))
        P_L = np.zeros(len(alpha_m))
        alpha_end_R = np.zeros(len(alpha_m))
        alpha_end_L = np.zeros(len(alpha_m))
        alpha_surf = np.zeros(len(alpha_m))

    ### Def asurf_s here
        asurf_1 = asurf_DFB
            
        #### Internal loss, alpha_i or alpha_fc, which can be added elsewhere
        alpha_fc = 0  
        #### Grating related loss. Jae Ha used alpha_fc = 0, so alpha_m is the grating related loss. Technically this would be the same as the threshold gain? (gth = 2alpham + alphai)
        #### However in this script, we're using alpha_i = alpha_w + alpha_bf, which is calculated in the plot DFB script during postprocessing, i.e. not here. So, alpha_grating_related
        #### is unused and alpha_fc = 0.    
        
        alpha_grating_related = 2*alpha_m + alpha_fc

        ### Declaring exports

        for i in range(len(alpha_m)):
            alpha0 = alpha_m[i]
            delta0 = delta_m[i]
            gt = alpha_grating_related[i]
            
            SL= 1 / np.sqrt(1 + np.abs(rL)**2 )
            S0 = SL * np.exp(0) #### replace with - 1j * phiL if using phase shift
            
            RL = rL * SL
            R0 = RL * np.exp(0) #### replace with 1j * phiL if using phase shift
            
            ### Lines 359 - 364, dropped DBR calc and using the end result which is S0 = Sd0, R0 = Rd0
            Rd0 = R0
            Sd0 = S0
            ### Zeta1 is the DFB zeta - keeping notation
            alpha01 = alpha0 - np.imag(zeta1)
            delta01 = delta0 + np.real(zeta1)
            gamma = gamma_DFB(kappa0, alpha01, delta01)
            
            
            ### Positions along the z axis in the device to calculate R, S at.
            z = np.linspace(0, 2*L1, num)

            Guided_Field = np.zeros(len(z))
            Near_Field = np.zeros(len(z))
            Near_Amp = np.zeros(len(z))
            Near_Phase = np.zeros(len(z))
            R = np.zeros(len(z))
            S = np.zeros(len(z))


            for j in range(len(z)):
                Guided_Field[j] = Guided(z[i], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                Near_Field[j] = Near2(z[j], R1, S1, S2, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)/(asurf_1**2)
                Near_Amp[j] = NearAmp(z[j], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                
                ### Computes the phase angle. Mimics line 424 of Finite_Solver_Trans_3d_sweep
                if len(z) % 2 == 0:
                    mid1 = NearAmp(z[len(z)//2 - 1], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                    mid2 = NearAmp(z[len(z)//2], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                    denom = (mid1 + mid2) / 2
                else:
                    denom = NearAmp(z[len(z)//2], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                Near_Phase[j] = np.angle(NearAmp(z[j], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1) / denom)
                R[j] = 1*Rg1(z[j], R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0)
                S[j] = 1*Sg1(z[j], S1, S2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0)
                
            ### Now moving onto the export calculation
            ni = 1
            ### Apparently in the original script they used Near2 twice? Once as a function and then again as a variable? Lmao
            ### Is the previous for loop redundant? Lines 530:
            Near_1 = np.zeros(num)
            Antisym1 = np.zeros(num)
            Guided1 = np.zeros(num)
            
            for j in range(len(z)):
                Near_1[j] = Near(z[j], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                Antisym1[j] = Antisym(z[j], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)
                Guided1[j] = Guided(z[j], S1, S2, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0, asurf_1)

            ### In Matlab, guided_1 = guided_2 ? 
            Guided2 = Guided1
            
            Gamma_lg[i] = 1 ### See line 552, Guided1 = Guided2

            alpha_surf[i] = asurf_1 * np.trapz(Near_1) * (L1 / num) / (np.trapz(Guided2) * (2 * L1 / num))
            alpha_end_R[i] = (np.abs(Rg1(2*L1, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0))**2) * (1 - np.abs(rR)**2) / (np.trapz(Guided2)*(2*L1)/num)
            alpha_end_L[i] = (np.abs(Sg1(0, S1, S2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0))**2)*(1-np.abs(rL)**2)/(np.trapz(Guided2)*(2*L1)/num)
            P_L[i] = (np.abs(Sg1(0, S1, S2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0))**2)*(1-np.abs(rL)**2)
            P_R[i] = (np.abs(Rg1(2*L1, R1, R2, L1, Lambda, kappa0, alpha01, delta01, gamma, Rd0, Sd0)))
            alpha_end[i] = (alpha_end_R[i]+alpha_end_L[i])/2
            Guided_export[i] = Guided_Field
            R_export[i,:] = R
            S_export[i,:] = S
            Z_export[i,:] = z
            

    return (alpha_m, delta_m, Gamma_lg, alpha_surf, Ratio_all, Guided_export, R_export, S_export, P_R, P_L)

def plot_mode_spectrum(k0, wavelength, Lambda, alpha_m, delta_m):

    # Assuming delta_m, lambda, Lambda, k0, and alpha_m are defined and calculated elsewhere in your script

    # Convert the given expressions to Python
    detuning_angstroms = (2 * np.pi / (k0 + delta_m / (wavelength / Lambda)) - wavelength) * 1e8

    # Plot 1: Alpha vs. Detuning in Angstroms
    plt.figure()
    plt.plot(detuning_angstroms, alpha_m, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    plt.xlabel('Detuning (Angstroms)')
    plt.ylabel('Alpha (cm^{-1})')
    plt.title('Mode Spectrum')
    plt.show()  # Display the plot

    # Plot 2: Threshold Gain vs. Detuning in Angstroms
    plt.figure()
    plt.plot(detuning_angstroms, alpha_m * 2 + 3.3, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    plt.xlabel('Detuning (Angstroms)')
    plt.ylabel('Threshold Gain, $g_{th}$ (cm^{-1})')
    plt.title('Mode Spectrum')
    plt.show()  # Display the plot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def Solve(i, j, k, L, l, wavelength, Lambda, derived_values, rR, rL, num_Modes, Plot_SWEEP, num_Z):
    k0 = 2 * np.pi / wavelength
    K0 = np.pi / Lambda
    
    ### Grabbing values
    deltak_DFB = derived_values['deltak_DFB'][i,j,k]
    alpha_DFB = derived_values['alpha_DFB'][i,j,k]
    deltak_DBR = derived_values['deltak_DBR'][i,j,k]
    alpha_DBR = derived_values['alpha_DBR'][i,j,k]
    asurf_DFB = derived_values['asurf_DFB'][i,j,k]
    asurf_DBR = derived_values['asurf_DBR'][i,j,k]
    kappa_DFB = derived_values['kappa'][i,j,k]
    zeta_DFB = derived_values['zeta'][i,j,k]
    
    alpha_extra_dbr = alpha_DBR - alpha_DFB
    
    DFB_Periods = round(L/(10*Lambda))
    #print(DFB_Periods)
    
    duty_cycle = derived_values['params'][2][k]
    #print(f"Duty cycle is {duty_cycle*100}%")
    
    L1 = DFB_Periods * Lambda / 2
    L2 = L1
    
    ### Inputs set in the model, passed into the function, not solved for.
    inputs = (kappa_DFB, zeta_DFB, L1, rR, rL, asurf_DFB, Lambda)


    ### Initial guess for alpha, deltak, which are solved for by fsolve
    ### Solving with Fvec_solv, i.e. transfer matrix method
    alpha_m, delta_m = Solver_Loop(inputs, num_Modes)
    
    print(f"Number of modes found: {len(alpha_m)}")
    
    #if ~np.isnan(alpha_m) == 0:
    #    print(f"No modes found, AAAAAAA!!!!!")
        
    ### Now, solving for the rest of the script with alpha0, delta as inputs to the coupled wave solver
        
    (alpha_m, delta_m, Gamma_lg, alpha_surf, Ratio_export, Guided_export, R_export, S_export, P_R, P_L) = Coupled_Wave_Solver(alpha_m, delta_m, inputs, num_Z)
    
    if Plot_SWEEP:
        plot_mode_spectrum(k0, wavelength, Lambda, alpha_m, delta_m)
    
    
    return (alpha_m, delta_m, Gamma_lg, alpha_surf, Ratio_export, Guided_export, R_export, S_export, P_R, P_L)
    