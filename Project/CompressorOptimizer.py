import numpy as np
import openmdao.api as om
from CompressorFirstStage import CompressorFirstStage
# from TurbineFirstStage import TurbineFirstStage
from CompressorStage import CompressorStage

#Takeoff Condition
M_flight = 0
T_flight = 288.15
P0_flight = 101325


# Cruise Condition
# M_flight = 0.8
# T_flight = 327.15
T0_flight = T_flight*(1+(1.4-1)*0.5*M_flight**2)
# P0_flight = 40297.5

global numStages, mdot, pr_c, T04_max

numStages = 8
mdot = 46  # kg/s
pr_c = 11  #Compressor stagnation ratio
T04_max = 1450  # K


class MultiStageCompressor(om.Group):

    def setup(self):
        compCycle = self.add_subsystem('compCycle', om.Group(), promotes=['*'])
        compCycle.add_subsystem('C1', CompressorFirstStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages'], promotes_outputs=['N'])
        for x in range(2, numStages + 1):
            compCycle.add_subsystem('C{0}'.format(x), CompressorStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages', 'N'])
            compCycle.connect('C{0}.T03'.format(x-1), 'C{0}.T01'.format(x))
            compCycle.connect('C{0}.P03'.format(x-1), 'C{0}.P01'.format(x))
            compCycle.connect('C{0}.alp3'.format(x-1), 'C{0}.alp1'.format(x))

class CompressorPerformance(om.ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('P_in', desc='Stag Pressure In')
        self.add_input('P_out', desc='Stag Pressure Out')
        self.add_input('T_in', desc='Stag T In')
        self.add_input('T_out', desc='Stag T Out')
        for i2 in range(1, numStages + 1):
            self.add_input('wdot{0}'.format(i2))

        self.add_output('n_overall')
        self.add_output('OPR')
        self.add_output('WdotTotal', desc="")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        P_in = inputs['P_in']
        P_out = inputs['P_out']
        T_in = inputs['T_in']
        T_out = inputs['T_out']
        wdots = []
        for i3 in range(1, numStages+1):
            wdots.append(inputs['wdot{0}'.format(i3)])

        T_out_isen = (P_out**0.285714*T_in)/(P_in**0.285714)
        n_overall = (T_out_isen-T_in)/(T_out-T_in)

        OPR = P_out/P_in

        outputs['OPR'] = OPR
        outputs['n_overall'] = n_overall
        outputs['WdotTotal'] = sum(wdots)







if __name__ == "__main__":
    prob = om.Problem()

    compCycle = prob.model.add_subsystem('compCycle', MultiStageCompressor(), promotes=['*'])

    # compCycle.add_subsystem('C1', CompressorFirstStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages'], promotes_outputs=['N'])
    #
    # for x in range(2, numStages + 1):
    #     compCycle.add_subsystem('C{0}'.format(x), CompressorStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages','N'])
    #     compCycle.connect('C{0}.T03'.format(x - 1), 'C{0}.T01'.format(x))
    #     compCycle.connect('C{0}.P03'.format(x - 1), 'C{0}.P01'.format(x))
    #     compCycle.connect('C{0}.alp3'.format(x - 1), 'C{0}.alp1'.format(x))

    prob.model.add_subsystem('CompressorPerformance', CompressorPerformance(), promotes_outputs=['n_overall', 'OPR'])
    prob.model.connect('C{0}.P03'.format(numStages), 'CompressorPerformance.P_out')
    prob.model.connect('C{0}.T03'.format(numStages), 'CompressorPerformance.T_out')


    # prob.model.add_subsystem('TurbineFirstStage', TurbineFirstStage(),
    #                          promotes_inputs=['*'],
    #                          promotes_outputs=['*'])


    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['maxiter'] = 500

    prob.model.add_design_var('cz', lower=1, upper=200)
    prob.model.add_design_var('C1.fCoeff', lower=0.1, upper=1)
    prob.model.add_design_var('C1.alp1', lower=np.deg2rad(0), upper=np.deg2rad(75.0))
    prob.model.add_design_var('rm', lower=0.1, upper=0.5)

    for i6 in range(1, numStages + 1):
        prob.model.add_design_var('C{0}.lCoeff'.format(i6), lower=0.1, upper=1)
        prob.model.add_design_var('C{0}.Df_r'.format(i6), lower=0.1, upper=0.55)
        prob.model.add_design_var('C{0}.sigma_s'.format(i6), lower=0.5, upper=2.0)
        prob.model.add_design_var('C{0}.chi3'.format(i6), lower=np.deg2rad(0), upper=np.deg2rad(75))
        prob.model.add_design_var('C{0}.camber_s'.format(i6), lower=np.deg2rad(0), upper=np.deg2rad(40))
        prob.model.add_design_var('C{0}.b_s'.format(i6), lower=0, upper=1)
        prob.model.add_design_var('C{0}.b_r'.format(i6), lower=0, upper=1)

        prob.model.add_constraint('C{0}.Df_s'.format(i6), lower=0.1, upper=0.55)
        prob.model.add_constraint('C{0}.M1_rel'.format(i6), lower=0.1, upper=0.75)
        prob.model.add_constraint('C{0}.M2'.format(i6), lower=0.1, upper=0.9)
        prob.model.add_constraint('C{0}.Pr_st'.format(i6), lower=0.1, upper=1.7)

        prob.model.connect('C{0}.Wdot'.format(i6), 'CompressorPerformance.wdot{0}'.format(i6))

    prob.model.add_constraint('C{0}.alp3'.format(numStages), lower=0, upper=0)
    prob.model.add_constraint('OPR', lower=pr_c, upper=pr_c)

    prob.model.add_objective('n_overall', scaler=-1)  # minimization
    prob.setup()


    # Compressor First Stage Inputs
    prob.set_val('rm', 0.7)
    prob.set_val('cz', 250)
    prob.set_val('C1.fCoeff', 0.3)
    prob.set_val('C1.lCoeff', 0.267)
    prob.set_val('C1.alp1', np.deg2rad(54))
    prob.set_val('C1.T01', T0_flight)
    prob.set_val('C1.P01', P0_flight)
    prob.set_val('CompressorPerformance.P_in', P0_flight)
    prob.set_val('CompressorPerformance.T_in', T0_flight)
    prob.set_val('mdot', mdot)
    prob.set_val('numStages', numStages)


    for i5 in range(1, numStages + 1):
        prob.set_val('C{0}.lCoeff'.format(i5), 0.3)
        prob.set_val('C{0}.Df_r'.format(i5), 0.5)
        prob.set_val('C{0}.sigma_s'.format(i5), 0.66)
        prob.set_val('C{0}.chi3'.format(i5), np.deg2rad(51.4))
        prob.set_val('C{0}.camber_s'.format(i5), np.deg2rad(11.7))
        prob.set_val('C{0}.thickness'.format(i5), 0.1)
        prob.set_val('C{0}.b_s'.format(i5), 0.1)
        prob.set_val('C{0}.b_r'.format(i5), 0.1)


    # prob.run_model()
    prob.run_driver()
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)




