import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import warnings
from Post_Process import plot_field_intensity
from Post_Process import plot_field_profile_real
from Transfer_Matrix_Model_Solver import Fr
from Transfer_Matrix_Model_Solver import Fp
class Parameters():
    def __init__(self, L, l, L_trans, rR, rL, Lambda, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR):
        self.L = L
        self.l = l
        self.L_trans = L_trans
        self.rR = rR 
        self.rL = rL
        self.Lambda = Lambda
        self.num_Z = num_Z
        self.cleave_phase_shift = cleave_phase_shift
        self.pi_phase_shift = pi_phase_shift
        self.theta = 0
        self.r_DFB_DBR = r_DFB_DBR
        self.L1 = 0
        self.L2 = 0
        self.l1 = 0
        self.l2 = 0
        self.Ll = self.L + self.l + self.L_trans * 2
        self.rLL = 0
        self.rLR = 0
        self.rRL = 0
        self.rRR = 0
        self.ztheta = 0
        self.asurf1 = 0
        self.asurf2 = 0
        self.asurf3 = 0
        self.alpha0 = 0
        self.delta = 0
        self.deltak_trans = 0
        self.alpha00 = 0
        self.alpha01 = 0
        self.alpha3 = 0
        self.kappa1 = 0
        self.kappa2 = 0
        self.kappa3 = 0
        self.delta0 = 0
        self.delta01 = 0
        self.delta3 = 0
        self.gamma_DBR = 0
        self.gamma_trans = 0
        self.gamma_DFB_ = 0
        self.zeta0 = 0
        self.zeta1 = 0
        self.zeta2 = 0
        ### Whoever wrote this in matlab is insane
        self.R0 = 0
        self.S0 = 0
        self.Rt10 = 0
        self.St10 = 0
        self.Rd0 = 0
        self.Sd0 = 0
        self.Rm = 0
        self.Sm = 0
        self.Rt20 = 0
        self.St20 = 0
        self.Rdm = 0
        self.Sdm = 0
        self.RL = 0
        self.SL = 0
        self.Ltot = 0
        self.zeta3 = 0
        self.delta_trans = 0

def F_DFB_matrix(kappa, alpha, delta, L):
    gamma = np.sqrt((kappa**2) + (alpha + 1j*delta)**2 ) 
    F = np.array([
        [np.cosh(gamma*L) + np.sinh(gamma*L)*(alpha + 1j*delta)/gamma, np.sinh(gamma*L)*1j*kappa/gamma],
        [np.sinh(gamma*L)*-1j*kappa/gamma, np.cosh(gamma*L) - np.sinh(gamma*L)*(alpha + 1j*delta)/gamma]
    ])
    return F

def Coupled_Wave_Solver_EEDFB(alpha_m, delta_m, kappa_DFB, zeta_DFB, L, l, L_trans, rR, rL, asurf_DFB, 
                              Lambda, num_Z, kappa_DBR, zeta_DBR, asurf_DBR, Gamma_ele, 
                              cleave_phase_shift, pi_phase_shift, r_DFB_DBR, plot_fields):
    
    params_ = Parameters(L, l, L_trans, rR, rL, Lambda, num_Z, cleave_phase_shift, pi_phase_shift, r_DFB_DBR)
    
    ### Gamma calc
    def gamma_DFB(kappa, alpha, delta):
        return np.sqrt((kappa**2) + (alpha + 1j*delta)**2 ) 

    ### Converting to relative frequency
    def alpha_prime(alpha, zeta):
        return alpha - np.imag(zeta)

    ### Converting to relative detuning
    def delta_prime(delta, zeta):
        return delta + np.real(zeta)


    ### Compiled coupled-mode solutions. Refactored from the Matlab script.
    def Sg(z, params):
        if z <= params.l1:
            return Fdfb_R(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.R0, params.S0, z)
        elif z <= params.l1 + params.L_trans:
            return Fdfb_S(params.kappa2, params.alpha3, params.delta3, params.gamma_trans, params.Rt10, params.St10, z - params.l1)
        elif z <= params.l1 + params.L_trans + params.L1:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, z - params.l1 - params.L_trans)
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, params.L1) * np.exp(-1j * 2 * np.pi / params.Lambda * (z - params.l1 - params.L_trans - params.L1))
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rm, params.Sm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta))
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans:
            return Fdfb_S(params.kappa2, params.alpha3, params.delta3, params.gamma_trans, params.Rt20, params.St20, z - (params.l1 + L_trans + params.L1 + params.ztheta + params.L2))
        else:
            return Fdfb_S(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.Rdm, params.Sdm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans))
        
    def Rg(z, params):
        if z <= params.l1:
            return Fdfb_R(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.R0, params.S0,z)
        elif z <= params.l1 + params.L_trans:
            return Fdfb_R(params.kappa3, params.alpha3, params.delta3, params.gamma_trans, params.Rt10, params.St10, z - params.l1)
        elif z <= params.l1 + params.L_trans + params.L1:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, z - params.l1 - params.L_trans)
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, params.L1) * np.exp(1j * 2 * np.pi / params.Lambda * (z - params.l1 - params.L_trans - params.L1))
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rm, params.Sm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta))
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans:
            return Fdfb_R(params.kappa3, params.alpha3, params.delta3, params.gamma_trans, params.Rt20, params.St20, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2))
        else:
            return Fdfb_R(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.Rdm, params.Sdm,z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans))
  
    def Sn(z, params):
        if z <= params.l1:
            return Fdfb_S(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.R0, params.S0, z) * params.asurf2
        elif z <= params.l1 + params.L_trans:
            return Fdfb_S(params.kappa2, params.alpha3, params.delta3, params.gamma_trans, params.Rt10, params.St10, z - params.l1) * params.asurf3
        elif z <= params.l1 + params.L_trans + params.L1:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, z - params.l1 - params.L_trans) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, params.L1) * np.exp(-1j * 2 * np.pi / params.Lambda * (z - params.l1 - params.L_trans - params.L1)) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2:
            return Fdfb_S(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rm, params.Sm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta)) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans:
            return Fdfb_S(params.kappa2, params.alpha3, params.delta3, params.gamma_trans, params.Rt20, params.St20, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2)) * params.asurf3
        else:
            return Fdfb_S(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.Rdm, params.Sdm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans)) * params.asurf2
        
    def Rn(z, params):
        if z <= params.l1:
            return Fdfb_R(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.R0, params.S0, z) * params.asurf2
        elif z <= params.l1 + params.L_trans:
            return Fdfb_R(params.kappa3, params.alpha3, params.delta3, params.gamma_trans, params.Rt10, params.St10, z - params.l1) * params.asurf3
        elif z <= params.l1 + params.L_trans + params.L1:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, z - params.l1 - params.L_trans) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rd0, params.Sd0, params.L1) * np.exp(1j * 2 * np.pi / params.Lambda * (z - params.l1 - params.L_trans - params.L1)) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2:
            return Fdfb_R(params.kappa1, params.alpha01, params.delta01, params.gamma_DFB_, params.Rm, params.Sm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta)) * params.asurf1
        elif z <= params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans:
            return Fdfb_R(params.kappa3, params.alpha3, params.delta3, params.gamma_trans, params.Rt20, params.St20, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2)) * params.asurf3
        else:
            return Fdfb_R(params.kappa2, params.alpha00, params.delta0, params.gamma_DBR, params.Rdm, params.Sdm, z - (params.l1 + params.L_trans + params.L1 + params.ztheta + params.L2 + params.L_trans)) * params.asurf2

    def Fdfb_R(kappa, alpha, delta, gamma, R, S, z):
        return (np.cosh(gamma*z) + np.sinh(gamma*z)*(alpha + 1j * delta) / gamma)*R + (1j*kappa*np.sinh(gamma*z)/gamma)*S

    def Fdfb_S(kappa, alpha, delta, gamma, R, S, z):
        return (np.cosh(gamma*z) - np.sinh(gamma*z)*(alpha + 1j * delta) / gamma)*S + (-1j*kappa*np.sinh(gamma*z)/gamma)*R

    Guided = lambda z, params: np.abs(Rg(z, params) ** 2) + np.abs(Sg(z, params) ** 2)
    Near = lambda z, params: np.abs(Sg(z, params) + Rg(z, params)) ** 2
    Near2 = lambda z, params: np.abs(Sn(z, params) + Rn(z, params)) ** 2
    Near_amp = lambda z, params: np.abs(Sn(z, params) + Rn(z, params))
    Antiysm = lambda z, params: np.abs(Sg(z, params) - Rg(z, params)) ** 2
    
    ### For export
    Guided_export = np.zeros((len(alpha_m), num_Z))
    Z_export = np.zeros((len(alpha_m), num_Z))
    R_export = np.zeros((len(alpha_m), num_Z))
    S_export = np.zeros((len(alpha_m), num_Z))
    Gamma_lg = np.zeros(len(alpha_m))
    alpha_end = np.zeros(len(alpha_m))
    P_R = np.zeros(len(alpha_m))
    P_L = np.zeros(len(alpha_m))
    alpha_end_R = np.zeros(len(alpha_m))
    alpha_end_L = np.zeros(len(alpha_m))
    alpha_surf = np.zeros(len(alpha_m))
    
    with warnings.catch_warnings():
        ### Ignoring the obnoxious warning about 'Casting complex values to reals'
        ### Will probably need a better fix for more precise designs 
        warnings.simplefilter("ignore", category = np.ComplexWarning)
        
        ### These were originally stored in a file and accessed directly, but then unused.
        ### Length z = 200, which is incompatible.
        alpha_trans = [0.2575, 0.2332]
        deltak_trans = 0
        params_.deltak_trans = deltak_trans
        asurf_trans = 0.0186
        kappa_trans = -0.1144 + 0.0122j
        zeta_trans = 1.4416 + 0.2453j
        ### Can move to the params .JSON file if needed.
        
        params_.kappa3 = kappa_trans * Gamma_ele
        params_.zeta3 = zeta_trans * Gamma_ele
        params_.asurf3 = asurf_trans * Gamma_ele
        params_.delta_trans = 0
        
        
        params_.DFB_Periods  = round(L/(10 * Lambda))
        params_.DBR_Periods  = round(l/(10 * Lambda))
        params_.L1 = L/2
        params_.L2 = L/2
        params_.l1 = l/2
        params_.l2 = l/2
        params_.Ltot = L + l + L_trans
        
        params_.kappa1 = kappa_DFB * Gamma_ele
        params_.zeta1 = zeta_DFB * Gamma_ele
        params_.asurf1 = asurf_DFB * Gamma_ele
        params_.kappa2 = kappa_DBR * Gamma_ele
        params_.zeta2 = zeta_DBR * Gamma_ele
        params_.asurf2 = asurf_DBR * Gamma_ele
        params_.kappa3 = zeta_trans * Gamma_ele
        
        ### This can be updated if needed, just generate a new file that calculates alpha_trans
        ### etc. with the correct length num_Z.
        
        ### Update this for the implementation of a DBR. Reflectivity of the transition from
        ### DBR to DFB (or DFB to DBR?)
        if r_DFB_DBR == 0:
            params_.rLL = 0
            params_.rLR = 0
            params_.rLRL = 0
            params_.rRR = 0
            params_.phiR = 0
        else:
            params_.phiL_DBR = np.random.rand() * np.pi * 2
            params_.phiR_DBR = np.random.rand() * np.pi * 2
            params_.rLL = np.random.rand() * params_.r_DFB_DBR * np.exp(1j * np.pi * 2 * np.random.rand())
            params_.rLR = np.random.rand() * params_.r_DFB_DBR * np.exp(1j * np.pi * 2 * np.random.rand())
            params_.rRL = np.random.rand() * params_.r_DFB_DBR * np.exp(1j * np.pi * 2 * np.random.rand())
            params_.rRR = np.random.rand() * params_.r_DFB_DBR * np.exp(1j * np.pi * 2 * np.random.rand())

        ### If there is a pi phase shift at the center of the cavity
        params_.theta = pi_phase_shift
        params_.ztheta = params_.theta * Lambda/ (2 * np.pi)
        params_.Ll = params_.Ll + params_.ztheta
            
        z = np.linspace(0, 2*params_.L1, params_.num_Z)

        for i in range(len(alpha_m)):
            
            params_.alpha0 = alpha_m[i]
            params_.delta = delta_m[i]
            
            params_.SL = 1 / np.sqrt(1+abs(params_.rL)**2)
            params_.S0 = params_.SL * np.exp(-1j * params_.cleave_phase_shift)
            params_.RL = params_.rL * params_.SL
            params_.R0 = params_.RL *np.exp(1j * params_.cleave_phase_shift)
            
            params_.alpha00 = -np.imag(params_.zeta2)
            params_.delta0 = params_.delta + np.real(params_.zeta2)      
            params_.alpha01 = params_.alpha0 - np.imag(params_.zeta1)
            params_.delta01 = params_.delta + np.real(params_.zeta1)
            params_.gamma_DBR = gamma_DFB(params_.kappa2, params_.alpha00, params_.delta0)
            params_.alpha3 = params_.alpha0 - np.imag(params_.zeta3)
            params_.delta3 = params_.delta + params_.delta_trans + np.real(params_.zeta3)
            
            ### Moving into the actual calculation. We calculate temporary field profiles for input into the next section, and then compile at the end.
            ### Temp fields (e.g. Rt20) are used in the intermediary calculations 
            ### "Trans section"
            RS_Trans_temp = Fr(params_.rLL) @ F_DFB_matrix(params_.kappa2, -np.imag(params_.zeta2), np.real(params_.zeta2) + params_.delta, params_.l1) @ Fp(params_.cleave_phase_shift) @ np.linalg.inv(Fr(params_.rL)) @ np.array([[params_.R0, params_.S0]]).T
            params_.Rt10 = RS_Trans_temp[0]
            params_.St10 = RS_Trans_temp[1]
            params_.gamma_trans = gamma_DFB(params_.kappa3, params_.alpha3, params_.delta3)   
            ###  section 1, 2    

            RS_DFB_temp = Fr(params_.rLR) @ np.linalg.inv(F_DFB_matrix(params_.kappa3, -np.imag(params_.zeta3) + params_.alpha0, np.real(params_.zeta3) + params_.delta + params_.deltak_trans, params_.L_trans)) @ Fr(params_.rLL) @ F_DFB_matrix(params_.kappa2, -np.imag(params_.zeta2), np.real(params_.zeta2) + params_.delta, params_.l1) @ Fp(params_.cleave_phase_shift) @ np.linalg.inv(Fr(params_.rL)) @  np.array([[params_.R0, params_.S0]]).T
            params_.Rd0 = RS_DFB_temp[0]
            params_.Sd0 = RS_DFB_temp[1]
            
            
            params_.gamma_DFB_ = gamma_DFB(params_.kappa1, params_.alpha01, params_.delta01)

            params_.Rm = Fdfb_R(params_.kappa1, params_.alpha01, params_.delta01, params_.gamma_DFB_, params_.Rd0, params_.Sd0, params_.L1) * np.exp(params_.pi_phase_shift*1j)
            params_.Sm = Fdfb_S(params_.kappa1, params_.alpha01, params_.delta01, params_.gamma_DFB_, params_.Rd0, params_.Sd0, params_.L1) * np.exp(-params_.pi_phase_shift*1j)
 
            RS_DFB_temp2 = Fr(params_.rRL) @ F_DFB_matrix(params_.kappa1, -np.imag(params_.zeta1) + params_.alpha0, np.real(params_.zeta1) + params_.delta, params_.L2) @ Fp(params_.pi_phase_shift) @ F_DFB_matrix(params_.kappa1, -np.imag(params_.zeta1) + params_.alpha0, np.real(params_.zeta1) + params_.delta, params_.L1) @ np.linalg.inv(Fr(params_.rLR)) @ F_DFB_matrix(params_.kappa3, -np.imag(params_.zeta3) + params_.alpha0, np.real(params_.zeta3) + params_.delta + params_.delta_trans, params_.L_trans) @ Fr(params_.rLL) @ F_DFB_matrix(params_.kappa2, -np.imag(params_.zeta2), np.real(params_.zeta2) + params_.delta, params_.l1) @ Fp(params_.cleave_phase_shift) @ np.linalg.inv(Fr(params_.rL)) @ np.array([[params_.R0, params_.S0]]).T
            params_.Rt20 = RS_DFB_temp2[0]
            params_.St20 = RS_DFB_temp2[1]
            
            RS_DBR_temp2 = Fr(params_.rRR) @ np.linalg.inv(F_DFB_matrix(params_.kappa3, -np.imag(params_.zeta3) + params_.alpha0, np.real(params_.zeta3) + params_.delta + params_.delta_trans, params_.L_trans)) @ Fr(params_.rRL) @ F_DFB_matrix(params_.kappa1, -np.imag(params_.zeta1) + params_.alpha0, np.real(params_.zeta1) + params_.delta, params_.L2) @ Fp(params_.pi_phase_shift) @F_DFB_matrix(params_.kappa1, -np.imag(params_.zeta1) + params_.alpha0, np.real(params_.zeta1) + params_.delta, params_.L1) @ np.linalg.inv(Fr(params_.rLR)) @ F_DFB_matrix(params_.kappa3, -np.imag(params_.zeta3) + params_.alpha0, np.real(params_.zeta3) + params_.delta + params_.delta_trans, params_.L_trans) @ Fr(params_.rLL) @ F_DFB_matrix(params_.kappa2, -np.imag(params_.zeta2), np.real(params_.zeta2) + params_.delta, params_.l1) @ Fp(params_.cleave_phase_shift) @ np.linalg.inv(Fr(params_.rL)) @ np.array([[params_.R0, params_.S0]]).T
            params_.Rdm = RS_DBR_temp2[0]
            params_.Sdm = RS_DBR_temp2[1]
            

            Guided_Field = np.zeros(len(z))
            Near_Field = np.zeros(len(z))
            Near_Amp = np.zeros(len(z))
            Near_Phase = np.zeros(len(z))
            R = np.zeros(len(z), dtype=np.complex_)
            S = np.zeros(len(z), dtype=np.complex_)

            for j in range(len(z)):
                zi = z[j]

                Guided_Field[j] = Guided(zi, params_)
                Near_Field[j] = Near2(zi, params_)
                Near_Amp[j] = Near_amp(zi, params_)
                R[j] = Rg(zi, params_)
                S[j] = Sg(zi, params_)

            Near_1 = np.zeros(len(z))
            Antisym1 = np.zeros(len(z))
            Guided1 = np.zeros(len(z))

            for j in range(len(z)):
                zi = z[j]
                Near_1[j] = Near(zi, params_)
                Antisym1[j] = Antiysm(zi, params_)
                Guided1[j] = Guided(zi, params_)
            
            Guided2 = Guided1

            alpha_surf[i] = params_.asurf1 * np.trapz(Near_1) * (params_.L1 / num_Z) / (np.trapz(Guided2) * (2 * params_.L1 / num_Z))
            alpha_end_R[i] = (np.abs(Rg(params_.L, params_))**2) * (1 - np.abs(params_.rR)**2) / (np.trapz(Guided2)*(params_.L)/num_Z)
            alpha_end_L[i] = (np.abs(Sg(0, params_))**2)*(1-np.abs(rL)**2)/(np.trapz(Guided2)*(params_.L)/num_Z)
            P_L[i] = (np.abs(Sg(0, params_))**2)*(1-np.abs(params_.rL)**2)
            P_R[i] = (np.abs(Rg(params_.L, params_)))
            alpha_end[i] = (alpha_end_R[i]+alpha_end_L[i])/2
            Guided_export[i] = Guided_Field
            R_export[i,:] = R
            S_export[i,:] = S
            Z_export[i,:] = z
            if i < 4:
                if plot_fields:
                    plot_field_intensity(z, Guided_Field, params_.L)                
                    plot_field_profile_real(z, R, S, params_.L)
                    plt.show()
    return (Gamma_lg, alpha_surf, Guided_export, R_export, S_export, 
            P_R, P_L, alpha_end, alpha_end_R, alpha_end_L, z)