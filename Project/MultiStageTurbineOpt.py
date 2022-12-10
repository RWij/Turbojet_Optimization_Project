# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 03:36:10 2022

@author: ryant

"""

import numpy as np
import openmdao.api as om
from TurbineFirstStage import TurbineFirstStage
from TurbineStage import TurbineStage

global numStages, T04_max, Po4

Po4 = 615000 #40297.5
numStages = 2
T04_max = 1450  # K
wdot_hpc = 14.6*10**6


class MultiStageTurbine(om.Group):

    def setup(self):
        turbCycle = self.add_subsystem('TurbCycle', om.Group(), promotes=['*'])
        turbCycle.add_subsystem('T1', TurbineFirstStage(), promotes_inputs=['mdot', 'N_rpms','wdot_hpc'], promotes_outputs=['rt', 'cz'])
        for x in range(2, numStages + 1):
            turbCycle.add_subsystem('T{0}'.format(x), TurbineStage(), promotes_inputs=['mdot', 'N_rpms', 'wdot_hpc', 'rt', 'cz'])
            turbCycle.connect('T{0}.To_exit'.format(x-1), 'T{0}.To4'.format(x))
            turbCycle.connect('T{0}.Po_exit'.format(x-1), 'T{0}.Po4'.format(x))
            turbCycle.connect('T{0}.alfa3'.format(x-1), 'T{0}.alfa1'.format(x))
            turbCycle.connect('T{0}.M_out'.format(x-1), 'T{0}.M4'.format(x))


class TurbinePerformance(om.ExplicitComponent):
    
    def initialize(self):
        pass

    def setup(self):
        self.add_input('Po4', desc='Inflow Stagnation Pressure of Turbine')
        self.add_input('Po5', desc='Exit Stagnation Pressure of Turbine')
        self.add_input('To4', desc='Inflow Stagnation Temperature of Turbine')
        self.add_input('To5', desc='Exit Stagnation Temperature of Turbine')
        self.add_input('gamma', 1.28, desc="Specific Heat Ratio")

        for i2 in range(1, numStages + 1):
            self.add_input('wdot{0}'.format(i2))

        self.add_output('n_overall')
        self.add_output('n_pt')
        self.add_output('wdotsum', desc="")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        Po4 = inputs['Po4']
        Po5 = inputs['Po5']
        To4 = inputs['To4']
        To5 = inputs['To5']
        gamma = inputs['gamma']

        wdots = []
        for i3 in range(1, numStages+1):
            wdots.append(inputs['wdot{0}'.format(i3)])

        T_out_isen = (Po5/Po4)**0.21875*To4
        n_overall = (T_out_isen - To4)/(To5 - To4)

        n_poly = (gamma / (gamma - 1)) / (np.log10(Po5 / Po4) / np.log10(To5 / To4))


        outputs['n_pt'] = n_poly
        outputs['n_overall'] = n_overall
        outputs['wdotsum'] = sum(wdots)


if __name__ == "__main__":
    prob = om.Problem()
    turbCycle = prob.model.add_subsystem('TurbCycle', MultiStageTurbine(), promotes=['*'])

    prob.model.add_subsystem('TurbinePerformance', TurbinePerformance(), \
                             promotes_outputs=['n_overall', 'n_pt'])
    prob.model.connect('T{0}.Po_exit'.format(numStages), 'TurbinePerformance.Po5')
    prob.model.connect('T{0}.To_exit'.format(numStages), 'TurbinePerformance.To5')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-3
    prob.driver.options['maxiter'] = 500


    for i6 in range(1, numStages + 1):
        prob.model.add_design_var('T{0}.Re_n'.format(i6), lower=0.1E5, upper=3.2E5)
        prob.model.add_design_var('T{0}.Re_r'.format(i6), lower=0.1E5, upper=3.2E5)
        prob.model.add_design_var('T{0}.DR'.format(i6), lower=0.1, upper=1.0)
        prob.model.add_design_var('T{0}.workpercent'.format(i6), lower=0.1, upper=1)
        #prob.model.add_design_var('T{0}.alfa2'.format(i6), lower=np.deg2rad(0), upper=np.deg2rad(75))
        prob.model.add_constraint('T{0}.M2_rel'.format(i6), lower=0.1, upper=1)
        prob.model.connect('T{0}.wdot_rotor'.format(i6), 'TurbinePerformance.wdot{0}'.format(i6))

    prob.model.add_constraint('T{0}.alfa3'.format(numStages), lower=0, upper=0)
    prob.model.add_constraint('TurbinePerformance.wdotsum'.format(numStages), lower=-wdot_hpc/0.99-wdot_hpc*0.01, upper=-wdot_hpc/0.99+wdot_hpc*0.01)

    prob.model.add_objective('n_overall', scaler=-1)  # minimization
    prob.setup()

    # combustor exit conditions
    prob.set_val('TurbinePerformance.Po4', Po4)
    prob.set_val('TurbinePerformance.To4', 1400.)

    prob.set_val('T1.alfa1', np.deg2rad(0))
    prob.set_val('T1.To4', 1436)
    prob.set_val('T1.Po4', 1096071)
    prob.set_val('T1.M4', 0.1464)
    prob.set_val('mdot', 46)
    prob.set_val('wdot_hpc', 14888767)
    prob.set_val('N_rpms', 6623)
    prob.set_val('T1.workpercent', 0.75955)
    prob.set_val('T2.workpercent', 0.2303)
    prob.set_val('T1.DR', 0.33165)
    prob.set_val('T2.DR', 0.1837)

    prob.set_val('T2.To4', 1248.4815)
    prob.set_val('T2.Po4', 543906)
    prob.set_val('rt', 0.38422)
    prob.set_val('cz', 106.66)
    prob.set_val('T2.alfa1', -1.27639)
    prob.set_val('T2.M4', 0.55168)
    prob.set_val('N_rpms', 6623)


    for i5 in range(1, numStages + 1):
        prob.set_val('T{0}.Re_n'.format(i5), 2.5E5)
        prob.set_val('T{0}.Re_r'.format(i5), 2.5E5)

    prob.run_model()
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)
