# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 10:04:40 2022

@author: ryant
"""

import numpy as np
from scipy import optimize
import openmdao.api as om
from CompressibleFlow import PR, TR, Area


class Diffuser(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('rd', 0.985)

    def setup(self):
        self.add_input("Ma", desc="Inlet Mach Number")
        self.add_input("mdot", desc="Mass Flow Rate")
        self.add_input("P0_flight", desc="Freestream Stagnation Pressure")
        self.add_input("T0_flight", desc="Freestream Stagnation Temperature")
        self.add_input("Aratio", desc="Inlet Area Ratio (inlet to outlet)")
        self.add_input('A_inlet', desc='Inlet Area')
        
        # self.add_output("A_inlet", 0.0, desc="Inlet Area")
        self.add_output("A_exit", 0.0, desc="Diffuser Exit Area")
        self.add_output("Po2", 0.0, desc="Diffuser Exit Stagnation Pressure")
        self.add_output("To2", 0.0, desc="Diffuser Exit Stagnation Temperature")
        self.add_output("M2", 0.0, desc="Diffuser Exit Mach Number")
        self.add_output("Cz2", 0.0, desc="Diffuser Exit Axial Velocity")
        self.add_output('ploss', desc="")
        
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-10)   
        
    def compute(self, inputs, outputs):
        gamma = 1.4
        R = 8314 / 29.80
        #Cp = (gamma * R) / (gamma - 1)
        Ma = inputs['Ma']
        P0_flight = inputs['P0_flight']
        T0_flight = inputs['T0_flight']
        Aratio = inputs['Aratio']
        A_inlet = inputs['A_inlet']
        mdot = inputs['mdot']

        rd = self.options['rd']

        Po2 = P0_flight * rd * np.power(1 + (0.5 * (gamma - 1) * Ma**2), (gamma - 1) / gamma)
        Pa = P0_flight / PR(Ma, gamma)
        To2 = T0_flight * np.power(Po2/Pa, (gamma - 1) / gamma)
        
        
        # A_inlet = Area(mdot, T0_flight, P0_flight, R, gamma, Ma)
        # print(A_inlet)
        rho_inlet = P0_flight/(R*T0_flight)
        cz1 = mdot/(rho_inlet*A_inlet)
        a1 = np.sqrt(gamma*R*T0_flight)
        M1 = cz1/a1
        
        Po2 = P0_flight * rd * np.power(1 + (0.5 * (gamma - 1) * M1**2), (gamma - 1) / gamma)
        Pa = P0_flight / PR(M1, gamma)
        To2 = T0_flight * np.power(Po2/Pa, (gamma - 1) / gamma)
        
        A_exit = A_inlet / Aratio
        # print(To2)
        def MachAreaFunc(x):
            Mm = x[0]
            A = 1 + ((gamma - 1) * 0.5 * M1 * M1)
            B = 1 + ((gamma - 1) * 0.5 * Mm * Mm)
            C = (gamma + 1) / (2 * (gamma - 1))
            return (((1/Aratio) * Po2/P0_flight * (1/M1) * np.power(A, C)) * Mm) - np.power(B, C)
        
        mach_rts = optimize.fsolve(MachAreaFunc, [0.0])

        M2 = mach_rts[0]
        
        T2 = To2/ TR(M2, gamma)
        Cz2 = M2 * np.sqrt(gamma * R * T2)

        ploss = Po2 -P0_flight

        outputs['ploss'] = ploss
        
        #outputs['A_inlet'] = A_inlet
        outputs['A_exit'] = A_exit
        outputs['Po2'] = Po2
        outputs['To2'] = To2
        outputs['M2'] = M2
        outputs['Cz2'] = Cz2

if __name__ == "__main__":
    prob = om.Problem()
    prob.model.add_subsystem('diffuser', Diffuser(), promotes=['*'])
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.model.add_objective('ploss')

    prob.model.add_design_var('Aratio', lower=0.1, upper =1)
    prob.model.add_design_var('A_inlet', lower=0.1, upper=2)
    prob.model.add_constraint('Cz2', lower=199.921, upper=199.921)

    prob.setup()


    # example values
    prob.set_val("Ma", 0.8)
    prob.set_val('mdot', 45.)
    prob.set_val('P0_flight', 35974.4)
    prob.set_val('T0_flight', 246)
    prob.set_val('A_inlet', 1.0)
    prob.set_val('Aratio', 0.95)

    prob.run_driver()
    
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)