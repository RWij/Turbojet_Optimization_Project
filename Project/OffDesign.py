# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 04:10:55 2022

@author: ryant
"""

from scipy import optimize
import numpy as np
from CompressibleFlow import TR
from Nozzle import Nozzle
from Combustor import Combustor
import openmdao.api as om

To4_od = 1220.

def OffDesign(n_pt_t, n_pt_c, N, mdot_d, M_d, To4_d, To4_od, To3_d, To2_od, To2_d, Ae):
    gamma = 1.4;
    R = 287.
    Cpc = (gamma * R) / (gamma - 1)
    
    gamma_t = 1.28
    R_t = R
    Cpt = (gamma_t * R_t) / (gamma_t - 1)
    
    delta_hr = 43.8E6
    
    # turbine temperature ratio (design point)
    Prc_d = 11 #To3_d / To2_d
    To3_To2_od = 1 + (((To4_od/To2_od)/(To4_d/To2_d)) * ((To3_d / To2_d) - 1))
    To3_od = To3_To2_od * To2_od
    print(To4_d/To2_d)
    print(To4_od/To2_od)
    a = np.sqrt((To4_d/To2_d)/(To4_od/To2_od))
    Prc_od = (To3_To2_od) ** ((gamma * n_pt_c) / (gamma - 1))
    print(Prc_od)
    mdot_od = mdot_d * (Prc_od / Prc_d) * a
    
    FAR_od = Cpc * (To4_od - To3_od) / (delta_hr - (Cpc * To4_od))
    To5_To4_od = 1 - (1 / (0.995 * (1 + FAR_od))) * (Cpc/Cpt) * (To2_od / To4_od) * ((To3_od / To2_od) - 1)
    print(To5_To4_od)
    Po5_Po4_od = To5_To4_od ** (gamma / ((gamma - 1) * n_pt_t))
    Prt_od = Po5_Po4_od
    
    # run combustor to get Po4
    prob = om.Problem()
    prob.model.add_subsystem('combustor', Combustor(), promotes_inputs=['DL_hi', 'Mz3', 'theta', 'Aratio_p', 'Aratio_d', 'rm', 'far', 'dg_hi', 'rd_DL', 'he'], promotes_outputs=['PLF_liner', 'delta_Po', 'lr_inner_3', 'lr_outer_3', 'Po4', 'L_total'])
    
    prob.setup()
    
    # input optimized combustor geometry
    prob.set_val('DL_hi', 3.0)
    prob.set_val('dg_hi', 0.8)
    prob.set_val('rd_DL', 0.4)
    prob.set_val('Mz3', 0.15)
    prob.set_val('theta', np.deg2rad(12.0))
    prob.set_val('Aratio_p', 0.1)
    prob.set_val('Aratio_d', 0.1)
    prob.set_val('far', 0.020)
    prob.set_val('rm', 0.025)
    prob.set_val('he', 0.035)

    prob.run_model() 
    
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)
    
    Po4_od = prob.get_val("Po4")
    Po5_od = Prt_od * Po4_od
    ##################################################
    
    def MachAreaFunc(x):
        M_od = x[0]
        A = (1 + ((gamma - 1) * 0.5 * M_d * M_d)) ** ((gamma + 1) / (2 * (1 - gamma)))
        B = 1 + ((gamma - 1) * 0.5 * M_od * M_od) ** ((gamma + 1) / (2 * (1 - gamma)))
        return (M_od*B) - ((M_d * A) * (mdot_od / mdot_d)) 
    mach_rts = optimize.fsolve(MachAreaFunc, [0.0])
    Me_diff_od = mach_rts[0]
    
    To2_T2 = TR(Me_diff_od, gamma)
    T2_od = To2_d / To2_T2
    To5_od = To5_To4_od * To4_od

    u = M_d * np.sqrt(gamma * R * T2_od)
    print(u)
    
    # run nozzle to get exit conditions
    prob = om.Problem()
    prob.model.add_subsystem('Nozzle', Nozzle(), promotes=['*'])
    
    prob.setup()
    
    # example values
    prob.set_val("To5", To5_od)
    prob.set_val('mdot', mdot_od)
    prob.set_val('Po5', Po5_od)
    prob.set_val('M5', Me_diff_od)
    
    prob.run_model()
    
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True) 
    
    ue = prob.get_val("ue")
    Pe_od = prob.get_val("Pe")
    ##################################################
    print(ue)
    thrust_od = mdot_od * (((1 + FAR_od) * ue) - u) + (Pe_od - 35974.4) * Ae
    
    out = {}
    out['mdot'] = mdot_od
    out['Me_diff'] = Me_diff_od
    out['RPM'] = N * a;
    out['FAR'] = FAR_od
    out['Prc'] = Prc_od
    out['Prt'] = Prt_od
    out['ST'] = thrust_od / mdot_od
    out['SFC'] = (FAR_od * mdot_od) / thrust_od
    out['Thrust'] = thrust_od
    return out
    

if __name__ == "__main__":
    print(OffDesign(0.83,\
              0.91,\
                  6623.3,\
                  46.,\
                  0.8,\
                  1436.5,\
                  1220,\
                  610.3,\
                  249.2,\
                  287.5,\
                  0.0956))