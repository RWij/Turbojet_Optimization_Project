# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 10:04:41 2022

@author: ryant
"""

import numpy as np
import openmdao.api as om
from CompressibleFlow import PR, TR, Area


class Nozzle(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_adia', default=0.95)
        self.options.declare('Me', default=1)
                
    def setup(self):
        # design variables
        self.add_input("To5", desc="Stagnation Turbine Exit Temperature")
        self.add_input("Po5", desc="Stagnation Turbine Exit Pressure")
        self.add_input("M5", desc="Turbine Exit Mach Number")
        self.add_input("mdot", desc="Mass Flow Rate into Nozzle")
        
        # design requirements
        self.add_output("Ae", desc="Fixed Converging Exit Nozzle Area")
        self.add_output("Pe", desc="Exit Nozzle Pressure")
        self.add_output("Te", desc="Exit Nozzle Temperature")
        self.add_output("ue", desc="Exit Nozzle Velocity")
                
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-10)   
        
    def compute(self, inputs, outputs):
        gamma = 1.35
        R = 8314 / 28.90
        Cp = (gamma * R) / (gamma - 1)
        To5 = inputs['To5']
        Po5 = inputs['Po5']
        M5 = inputs['M5']
        mdot = inputs['mdot']
        
        n_adia = self.options['n_adia']
        Me = self.options['Me']
                        
        ue = np.sqrt(2 * Cp * To5 * (1 - 1/(1 + (gamma - 1) * 0.5 * Me**2)))
        
        Poe_Pe = PR(Me, gamma)
        #Toe_Te = TR(Me, gamma)
        #Toe = To5 * n_adia
        Te_Toe = 1 - (n_adia * (1 - (1/Poe_Pe)**((gamma-1)/gamma)))
        Toe = Te_Toe * To5
        Poe = Po5 # assuming negligible losses
        Pe = Poe / Poe_Pe
        #Te = Toe / Toe_Te
        Te = Toe * Te_Toe
        
        A = 1 + ((gamma - 1) * 0.5 * (Me**2))
        B = 1 + ((gamma - 1) * 0.5 * (M5**2))

        A5 = Area(mdot, To5, Po5, R, gamma, M5)
        #print(A5)
        Ae_A5 = (Po5 / Poe) * (M5 / Me) * np.power(A / B, (gamma + 1)/(2*(gamma - 1)))
        #print(Ae_A5)
        Ae = Ae_A5 * A5
                
        outputs['Ae'] = Ae
        outputs['Pe'] = Pe
        outputs['Te'] = Te
        outputs['ue'] = ue
        

if __name__ == "__main__":
    prob = om.Problem()
    prob.model.add_subsystem('Nozzle', Nozzle(), promotes=['*'])
    
    prob.setup()
    
    # example values
    prob.set_val("To5", 1191)
    prob.set_val('mdot', 46)
    prob.set_val('Po5', 416543)
    prob.set_val('M5', 0.16076291)

    prob.run_model()
    
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)        
        