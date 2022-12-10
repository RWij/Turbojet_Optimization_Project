# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 04:19:49 2022

@author: ryant
"""

import numpy as np
import openmdao.api as om
from scipy import optimize
from CompressibleFlow import PR, TR, Area, RhoR


class Combustor(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_comb', default=0.99)
        self.options.declare('delta_hr', default=43.8*10**6)
        self.options.declare('snout_frac', default=0.6)
        self.options.declare('T_res', default=3.5*10**-3)
        self.options.declare('rmi_rme', default=1)

    def setup(self):
        # design variables
        self.add_input('theta', desc="Prediffuser Angle")
        self.add_input('dg_hi', desc="Dump Gap Ratio")
        self.add_input('rd_DL', desc="Dome Corner Radius Ratio")
        self.add_input('DL_hi', desc="Liner Depth Ratio")
        self.add_input('Mz3', desc="Pre-combustion Mach Number")
        self.add_input('Aratio_d', desc="Diffuser inlet to Max CS Area")
        self.add_input('Aratio_p', desc="Pre-diffuser area ratio (inlet / exit)")
        self.add_input('he', desc="Prediffuser Outlet height")
        self.add_input('rm', desc="Diffuser Inlet Height")
        self.add_input('far', desc="Fuel-to-Air Ratio")
        
        # constraints/optimization
        self.add_output('PLF_liner', desc="Cold Liner Pressure Loss Factor")
        self.add_output('delta_Po', desc="Statnation Pressure Loss")
        self.add_output('L_total', desc="Total Combustor Length")

        # given assumption:
        self.add_input('delta_hr', desc="Fuel Heating Value")
        self.add_input('To3', desc="Temperature into Combustor")
        self.add_input('mdot', desc="Mass Flow Rate")
        self.add_input('Po3', desc="Pressure into Combustor")
        
        # for design report
        self.add_output('L_diff', desc="Length of Diffuser")
        self.add_output('L_fl_tube', desc="Length of Flame Tube")
        self.add_output('lr_inner_3', desc="Liner Inner (Min) Radius")
        self.add_output('lr_m_3', desc="Liner Mean Radius")
        self.add_output('lr_outer_3', desc="Liner Outer (Max) Radius")
        self.add_output('lM3', desc="Diffuser Exit Mach Number")
        self.add_output('D_L', desc="Line Depth")
        self.add_output('A_h_eff', desc="Effective Flow Area")
        self.add_output('To4', desc="Combustor Exit Temperature")
        self.add_output('Po4', desc="Combustor Exit Pressure")
        self.add_output('Po4_Po3', desc="Rotor Hub Radius")
        
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-10)   

    def compute(self, inputs, outputs):
        gamma = 1.4
        MW_air= 28.90
        R = 8314 / MW_air
        Cp = gamma * R / (gamma - 1)
        
        Aratio_d = inputs['Aratio_d']
        Aratio_p = inputs['Aratio_p']
        M = inputs['Mz3']
        To3 = inputs['To3']
        Po3 = inputs['Po3']
        theta = inputs['theta']
        he = inputs['he']
        rm = inputs['rm']
        dg_hi = inputs['dg_hi']
        rd_DL = inputs['rd_DL']
        DL_hi = inputs['DL_hi']
        mdot = inputs['mdot']
        far = inputs['far']

        n_comb = self.options['n_comb']
        delta_hr = self.options['delta_hr']
        snout_frac = self.options['snout_frac']
        T_res = self.options['T_res']
        rmi_rme = self.options['rmi_rme']


        T3 = To3 / TR(M, gamma)
        Cz = M * np.sqrt(gamma * R * T3)
        
        Kt = np.power(1 - Aratio_d, 2) + np.power(1 - Aratio_d, 6)
        Po3m_Po3 = np.exp(-0.5 * gamma * M * M * Kt)
        
        Po3m = Po3m_Po3 * Po3
        A3 = Area(mdot * snout_frac, To3, Po3, R, gamma, M)
        Aref_dif = Area(mdot, To3, Po3, R, gamma, M)
        Aref = A3 / Aratio_p
        rho = mdot / (Cz * A3)
        
        # for cold liner
        C_ref = mdot / (rho * Aref_dif)
        q_ref = 0.5 * rho * (C_ref ** 2)
        PLF_liner = (Po3 - Po3m) / q_ref
        #PLF_liner = (abs(Po3m - Po3)) / (((0.5 * gamma * M**2) / PR(M, gamma)) * np.power(Aratio_d, 2))
        
        Po3m = Po3m_Po3 * Po3
        delta_Po = Po3m - Po3 # should be negative
        
        hi = he * Aratio_p / rmi_rme
        DL = DL_hi * hi
        

        A_h_eff = np.sqrt(1 / PLF_liner) * Aref
                
        def MachAreaFunc(x):
            Mm = x[0]
            A = 1 + ((gamma - 1) * 0.5 * M * M)
            B = 1 + ((gamma - 1) * 0.5 * Mm * Mm)
            C = (gamma + 1) / (2 * (gamma - 1))
            return (((1/Aratio_d) * Po3m_Po3 * (1/M) * np.power(A, C)) * Mm) - np.power(B, C)
        
        mach_rts = optimize.fsolve(MachAreaFunc, [0.0])

        lM3 = mach_rts[0]
        
        rho_om = Po3m / (R * To3)
        rho_m = rho_om / RhoR(lM3, gamma)
        
        D_o_plus_D_i = (Aref / (2 * 3.14 * rm)) - DL
        h3i = DL + D_o_plus_D_i 
        lr_inner_3 = rm - (hi / 2)
        lr_m_3 = Aref / (2 * 3.14 * h3i)
        lr_outer_3 = rm + (hi / 2)
        
        L_pre = (he - hi) / (2 * np.tan(theta))
        L_dump = (dg_hi * hi) + (rd_DL * DL)
        L_diff = L_pre + L_dump
        
        L_fl_tube = (3 * mdot * T_res) / (rho_m * Aref)
        
        L_total = L_diff + L_fl_tube
        
        To4 = (((far * delta_hr) + (Cp * To3)) * n_comb) / (Cp * (1 + far))
        
        #Po4 = Po3m - ((PLF_liner * (mdot**2)/(2 * rho * (Aref**2))))
        Po4 = Po3m
        #Po4 = ((To4 / To3) ** (-0.5 * gamma * lM3))
        
        Po4_Po3 = Po4 / Po3
                
        outputs['PLF_liner'] = PLF_liner
        outputs['delta_Po'] = delta_Po
        
        outputs['L_diff'] = L_diff
        outputs['L_fl_tube'] = L_fl_tube
        outputs['L_total'] = L_total
        outputs['lr_inner_3'] = lr_inner_3
        outputs['lr_m_3'] = lr_m_3
        outputs['lr_outer_3'] = lr_outer_3
        outputs['lM3'] = lM3
        outputs['D_L'] = DL_hi * hi
        outputs['A_h_eff'] = A_h_eff
        outputs['To4'] = To4
        outputs['Po4'] = Po4
        outputs['Po4_Po3'] = Po4_Po3
        
    
if __name__ == "__main__":
    prob = om.Problem()
    prob.model.add_subsystem('combustor', Combustor(), promotes=['*'])
    
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    
    prob.model.add_design_var('DL_hi', lower=2.5, upper=6.0)
    prob.model.add_design_var('dg_hi', lower=0.8, upper=1.2)
    prob.model.add_design_var('rd_DL', lower=0.25, upper=0.5)
    prob.model.add_design_var('theta', lower=np.deg2rad(7.0), upper=np.deg2rad(21.0))
    prob.model.add_design_var('Mz3', lower=0.1, upper=0.3)
    prob.model.add_design_var('Aratio_p', lower=0.1, upper=0.8)
    prob.model.add_design_var('Aratio_d', lower=0.1, upper=0.8)
    prob.model.add_design_var('rm', lower=0.01, upper=0.03)
    prob.model.add_design_var('far', lower=0.016, upper=0.040)
    prob.model.add_design_var('he', lower=0.01, upper=0.05)
    
    prob.model.add_constraint('PLF_liner', lower=5.0, upper=5.0)
    prob.model.add_constraint('lr_inner_3', lower=0.01, upper=0.1)
    prob.model.add_constraint('lr_outer_3', lower=0.01, upper=0.1)
    prob.model.add_constraint('To4', lower=900., upper=1450.)
    
    prob.model.add_objective('delta_Po', scaler=-1)
    
    prob.setup()
    
    prob.set_val('DL_hi', 3.0)
    prob.set_val('dg_hi', 0.8)
    prob.set_val('rd_DL', 0.4)
    prob.set_val('Mz3', 0.15)
    prob.set_val('theta', np.deg2rad(12.0))
    prob.set_val('Aratio_p', 0.5)
    prob.set_val('Aratio_d', 0.5)
    prob.set_val('far', 0.020)
    prob.set_val('rm', 0.025)
    prob.set_val('he', 0.035)

    prob.set_val('To3', 610)
    prob.set_val('Po3', 1114568)
    prob.set_val('Mz3', 0.41)

    #prob.run_model()
    results = prob.run_driver()
    
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)
