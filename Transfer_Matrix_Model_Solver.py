import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sc
from scipy.optimize import fsolve
import numpy as np
from Post_Process import plot_mode_spectrum
from Post_Process import plot_field_intensity

### Gamma calc
def gamma_DFB(kappa, alpha, delta):
    return np.sqrt((kappa**2) + (alpha + 1j*delta)**2 ) 

### Converting to relative frequency
def alpha_prime(alpha, zeta):
    return alpha - np.imag(zeta)

### Converting to relative detuning
def delta_prime(delta, zeta):
    return delta + np.real(zeta)

### Basic structure of the finite mode structure component F_DFB
### 1 S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003)
def F_DFB_matrix(alpha, delta, kappa, gamma, L):
    F = np.array([
        [np.cosh(gamma*L) + (alpha + 1j*delta)/gamma*np.sinh(gamma*L), 1j*kappa/gamma*np.sinh(gamma*L)],
        [-1j*kappa/gamma*np.sinh(gamma*L), np.cosh(gamma*L) - (alpha + 1j*delta)/gamma*np.sinh(gamma*L)]
    ])
    return F

### For facet reflectivity
def Fr(r):
    inv_denom = 1 / (1 - r)
    return np.array([
        [inv_denom, -r * inv_denom],
        [-r * inv_denom, inv_denom]
    ])
    
### For pi phase shift, currently unused
def Fp(phi):
    return [
        [np.exp(1j*phi), 0], 
        [0, np.exp(-1j*phi)]] 
    
### 1 S. Li et al., IEEE J. Sel. Topics. Quantum Electron. 9, 1153 (2003)

### UPDATE for the Cleave location shift 
def F_DFB(alpha, delta, kappa0, zeta0, L0, rR, rL, asurf, Lambda):

    alpha_prime_DFB = alpha_prime(alpha, zeta0) ### Converting to alpha prime, delta prime
    delta_prime_DFB = delta_prime(delta, zeta0)
    gamma = gamma_DFB(kappa0, alpha_prime_DFB, delta_prime_DFB)
          
    #### F_DFB_matrix Input = [kappa, gamma, L]

    F = Fr(rR) @ F_DFB_matrix(alpha_prime_DFB, delta_prime_DFB, kappa0, gamma, L0) @ F_DFB_matrix(alpha_prime_DFB, delta_prime_DFB, kappa0, gamma, L0) @ np.linalg.inv(Fr(rL))
    return F

### See Li 2003 - 
### Essentially, the lasing condition is that with no incoming light, we still generate outcoming light. F22 is returned as the condition
def F22(F):
    return F[1, 1] 
    
### Solver base
def Fvec_solv(x, *args):
    alpha, delta = x
    F = F_DFB(alpha, delta, *args)
    F22_value = F22(F)  
    return [np.real(F22_value), np.imag(F22_value)]

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
            guess = [j / L, i / L - np.real(zeta0)]
            guess_all.append(guess)
            norm_guess = np.linalg.norm(Fvec_solv(guess, *inputs))
            norm_guess_all.append(norm_guess)
            ### Converting to mm^-1 (eventually convert everything to cm)
            min_Fval = max(300, np.abs(kappa0)**2 * 10)
            if not alpha0 or guess[0] < np.min(alpha0):
                min_Fval *= 100
            
            ### If the norm guess of Fvec_solv is less than the max of 300, np.abs(kappa0)**2 * 10 (300 is arbitrary?) continue
            if norm_guess < min_Fval:
                solution, info, ier, mesg = fsolve(Fvec_solv, guess, args=inputs, full_output=True, xtol=1E-20, epsfcn=1E-20)

                ### If alpha0 is empty, or if the functions are not close enoguh to the target
                ### "In summary, the expression is used to determine if the current solution found by fsolve is essentially a duplicate of any solution found in previous iterations, based on a specified tolerance (0.01). If it returns True, the current solution is very close to at least one previously found solution."
                if not alpha0 or not np.any(np.all(np.abs(np.vstack([alpha0, delta]).T - solution) < 0.01, axis=1)):
                    alpha0.append(solution[0])
                    delta.append(solution[1])
                
                ### If not empty
                elif alpha0:
                    index = int(np.where(np.prod(np.abs(np.vstack([alpha0, delta]).T - solution) < 0.01, axis=1))[0])

                    if np.abs(Fvec_solv(solution, *inputs))[0] < np.abs(Fvec_solv([alpha0[index], delta[index]], *inputs))[0]:
                        ### Then store the new results
                        alpha0[index] = solution[0]
                        delta[index] = solution[1]
                        
        ### j = num modes found for this iteration
        if len(alpha0) > 2 and np.max(alpha0) < (j - 20) / L:
            break
            
    if alpha0:
        sorted_indices = np.argsort(alpha0)
        alpha0 = np.array(alpha0)[sorted_indices]
        delta = np.array(delta)[sorted_indices]
        num_modes = ~np.isnan(alpha0)
        alpha0 = alpha0[num_modes]
        delta = delta[num_modes]         
    return alpha0, delta

def plot_mode_spectrum(k0, wavelength, Lambda, alpha_m, delta_m):

    detuning_angstroms = (2 * np.pi / (k0 + delta_m / (wavelength / Lambda)) - wavelength) * 1e8

    plt.figure()
    plt.plot(detuning_angstroms, alpha_m, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    plt.xlabel('Detuning (Angstroms)')
    plt.ylabel('Alpha (cm^{-1})')
    plt.title('Mode Spectrum')
    plt.show()  

    plt.figure()
    plt.plot(detuning_angstroms, alpha_m * 2 + 3.3, 'o', markerfacecolor='b', markeredgecolor='b', markersize=10)
    plt.xlabel('Detuning (Angstroms)')
    plt.ylabel('Threshold Gain, $g_{th}$ (cm^{-1})')
    plt.title('Mode Spectrum')
    plt.show()  

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
    kappa_DFB = derived_values['kappa_DFB'][i,j,k]
    zeta_DFB = derived_values['zeta_DFB'][i,j,k]
    kappa_DBR = derived_values['kappa_DBR'][i,j,k]
    zeta_DBR = derived_values['zeta_DBR'][i,j,k]
    
    alpha_extra_dbr = alpha_DBR - alpha_DFB
    DFB_Periods = round(L/(Lambda))
    duty_cycle = derived_values['params'][2][k]
    L1 = DFB_Periods * Lambda / 2
    L2 = L1

    ### Inputs set in the model, passed into the function, not solved for.
    inputs = (kappa_DFB, zeta_DFB, L1, rR, rL, asurf_DFB, Lambda)

    ### Initial guess for alpha, deltak, which are solved for by fsolve
    ### Solving with Fvec_solv, i.e. transfer matrix method
    alpha_m, delta_m = Solver_Loop(inputs, num_Modes)    
    print(f"Number of modes found: {len(alpha_m)}")

    
    if Plot_SWEEP:
        plot_mode_spectrum(k0, wavelength, Lambda, alpha_m, delta_m)
        
    return(alpha_m, delta_m, kappa_DFB, zeta_DFB, asurf_DFB, kappa_DBR, zeta_DBR, asurf_DBR)
