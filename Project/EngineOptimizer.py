import numpy as np
import openmdao.api as om
from CompressorFirstStage import CompressorFirstStage
# from TurbineFirstStage import TurbineFirstStage
from CompressorStage import CompressorStage
from Combustor import Combustor
from CompressorOptimizer import CompressorPerformance
from CompressorOptimizer import MultiStageCompressor
from IGV import InletGuideVane
from Diffuser import Diffuser
from TurbineFirstStage import TurbineFirstStage
from TurbineStage import TurbineStage
from MultiStageTurbineOpt import TurbinePerformance
from Nozzle import Nozzle

#Takeoff Condition
M_flight = 0
T_flight = 288.15
P0_flight = 101325


# Cruise Condition
# M_flight = 0.8
# T_flight = 327.15
T0_flight = T_flight*(1+(1.4-1)*0.5*M_flight**2)
# P0_flight = 40297.5

global numStages, numStagesTurbine, mdot, pr_c, T04_max, wdot_hpc

numStages = 8
numStagesTurbine = 2
mdot = 46  # kg/s
pr_c = 11  #Compressor stagnation ratio
T04_max = 1450  # K
wdot_hpc = 14.6*10**6


class Engine(om.Group):

    def setup(self):
        engineCycle = self.add_subsystem('engineCycle', om.Group(), promotes=['*'])

        # engineCycle.add_subsystem('Diffuser', Diffuser(), promotes_inputs=[], promotes_outputs=[])
        engineCycle.add_subsystem('IGV', InletGuideVane(), promotes_inputs=[], promotes_outputs=[])
        engineCycle.add_subsystem('Compressor', MultiStageCompressor(), promotes_inputs=[], promotes_outputs=[])
        engineCycle.add_subsystem('Combustor', Combustor(), promotes_inputs=[], promotes_outputs=[])
        # engineCycle.add_subsystem('Turbine', MultiStageTurbine(), promotes_inputs=[], promotes_outputs=[])
        # engineCycle.add_subsystem('Nozzle', Nozzle(), promotes_inputs=[], promotes_outputs=[])


if __name__ == "__main__":
    prob = om.Problem()
    engine = prob.model.add_subsystem('engine', om.Group(), promotes=['*'])
    # engine.nonlinear_solver = newton = om.NonlinearBlockGS()

    #Diffuser Setup
    prob.model.engine.add_subsystem('diffuser', Diffuser(), promotes_inputs=['mdot','P0_flight', 'T0_flight'])

    engine.connect('diffuser.Po2', 'compressor.IGV.P0in')
    engine.connect('diffuser.Cz2', 'compressor.IGV.czin')
    # Compressor Setup

    prob.model.engine.add_subsystem('compressor', om.Group(), promotes_inputs=['mdot'])

    prob.model.engine.compressor.add_subsystem('IGV', InletGuideVane(), promotes_inputs=['mdot', 'rm', 'alp1', 'cz'])
    prob.model.engine.compressor.add_subsystem('C1', CompressorFirstStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages', 'alp1'], promotes_outputs=['N'])

    for x in range(2, numStages + 1):
        prob.model.engine.compressor.add_subsystem('C{0}'.format(x), CompressorStage(), promotes_inputs=['rm', 'cz', 'mdot', 'numStages', 'N'])
        engine.connect('compressor.C{0}.T03'.format(x - 1), 'compressor.C{0}.T01'.format(x))
        engine.connect('compressor.C{0}.P03'.format(x - 1), 'compressor.C{0}.P01'.format(x))
        engine.connect('compressor.C{0}.alp3'.format(x - 1), 'compressor.C{0}.alp1'.format(x))

    prob.model.engine.compressor.add_subsystem('CompressorPerformance', CompressorPerformance(), promotes_outputs=['n_overall', 'OPR'])
    engine.connect('compressor.C{0}.P03'.format(numStages), 'compressor.CompressorPerformance.P_out')
    engine.connect('compressor.C{0}.T03'.format(numStages), 'compressor.CompressorPerformance.T_out')

    # IGV Setup
    engine.connect('compressor.IGV.T0out', 'compressor.C1.T01')
    engine.connect('compressor.IGV.P0out', 'compressor.C1.P01')
    engine.connect('compressor.IGV.P0out', 'compressor.CompressorPerformance.P_in')
    engine.connect('compressor.IGV.T0out', 'compressor.CompressorPerformance.T_in')

    # Combustor Setup
    prob.model.engine.add_subsystem('combustor', Combustor(), promotes_inputs=['mdot'])
    engine.connect('compressor.C{0}.P03'.format(numStages), 'combustor.Po3')
    engine.connect('compressor.C{0}.T03'.format(numStages), 'combustor.To3')
    engine.connect('compressor.C{0}.M_out'.format(numStages), 'combustor.Mz3')
    # TODO: Might need to connect compressor rm to combustor rm
    # engine.connect('compressor.rm', 'combustor.rm')


    # Turbine Setup

    prob.model.engine.add_subsystem('turbine', om.Group(), promotes_inputs=['mdot'])

    prob.model.engine.turbine.add_subsystem('T1', TurbineFirstStage(), promotes_inputs=['mdot', 'N_rpms', 'wdot_hpc', 'To4', 'Po4', 'M4'],
                                            promotes_outputs=['rt', 'cz'])
    for i7 in range(2, numStagesTurbine + 1):
        prob.model.engine.turbine.add_subsystem('T{0}'.format(i7), TurbineStage(),
                                                promotes_inputs=['mdot', 'N_rpms', 'wdot_hpc', 'rt', 'cz'])
        engine.connect('turbine.T{0}.To_exit'.format(i7 - 1), 'turbine.T{0}.To4'.format(i7))
        engine.connect('turbine.T{0}.Po_exit'.format(i7 - 1), 'turbine.T{0}.Po4'.format(i7))
        engine.connect('turbine.T{0}.alfa3'.format(i7 - 1), 'turbine.T{0}.alfa1'.format(i7))
        engine.connect('turbine.T{0}.M_out'.format(i7 - 1), 'turbine.T{0}.M4'.format(i7))

    prob.model.engine.turbine.add_subsystem('TurbinePerformance', TurbinePerformance(), promotes_outputs=['n_overall', 'n_pt'])
    engine.connect('turbine.T{0}.Po_exit'.format(numStagesTurbine), 'turbine.TurbinePerformance.Po5')
    engine.connect('turbine.T{0}.To_exit'.format(numStagesTurbine), 'turbine.TurbinePerformance.To5')

    # Nozzle
    prob.model.engine.add_subsystem('nozzle', Nozzle(), promotes_inputs=['mdot'])

    engine.connect('turbine.T{0}.To_exit'.format(numStagesTurbine), 'nozzle.To5')
    engine.connect('turbine.T{0}.Po_exit'.format(numStagesTurbine), 'nozzle.Po5')
    engine.connect('turbine.T{0}.M_out'.format(numStagesTurbine), 'nozzle.M5')

    prob.setup()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['maxiter'] = 500

    # Diffuser
    prob.model.add_design_var('diffuser.Aratio', lower=0.1, upper =1)
    prob.model.add_design_var('diffuser.A_inlet', lower=0.1, upper=2)

    # Compressor

    # Compressor Design Variables
    prob.model.add_design_var('compressor.cz', lower=1, upper=200)
    prob.model.add_design_var('compressor.C1.fCoeff', lower=0.1, upper=1)
    prob.model.add_design_var('compressor.alp1', lower=np.deg2rad(0), upper=np.deg2rad(75.0))
    prob.model.add_design_var('compressor.rm', lower=0.1, upper=0.5)

    for i6 in range(1, numStages + 1):
        prob.model.add_design_var('compressor.C{0}.lCoeff'.format(i6), lower=0.1, upper=1)
        prob.model.add_design_var('compressor.C{0}.Df_r'.format(i6), lower=0.1, upper=0.55)
        prob.model.add_design_var('compressor.C{0}.sigma_s'.format(i6), lower=0.5, upper=2.0)
        prob.model.add_design_var('compressor.C{0}.chi3'.format(i6), lower=np.deg2rad(0), upper=np.deg2rad(75))
        prob.model.add_design_var('compressor.C{0}.camber_s'.format(i6), lower=np.deg2rad(0), upper=np.deg2rad(40))
        prob.model.add_design_var('compressor.C{0}.b_s'.format(i6), lower=0, upper=1)
        prob.model.add_design_var('compressor.C{0}.b_r'.format(i6), lower=0, upper=1)

        # Compressor Constraints
        prob.model.add_constraint('compressor.C{0}.Df_s'.format(i6), lower=0.1, upper=0.55)
        prob.model.add_constraint('compressor.C{0}.M1_rel'.format(i6), lower=0.1, upper=0.75)
        prob.model.add_constraint('compressor.C{0}.M2'.format(i6), lower=0.1, upper=0.9)
        prob.model.add_constraint('compressor.C{0}.Pr_st'.format(i6), lower=0.1, upper=1.7)

        prob.model.connect('compressor.C{0}.Wdot'.format(i6), 'compressor.CompressorPerformance.wdot{0}'.format(i6))

    prob.model.add_constraint('compressor.C{0}.alp3'.format(numStages), lower=0, upper=0)
    prob.model.add_constraint('compressor.OPR', lower=pr_c, upper=pr_c)

    # IGV
    # prob.model.add_design_var('compressor.IGV.czin', lower=1, upper=300)
    prob.model.add_design_var('compressor.IGV.b', lower=0.01, upper=1)

    # Combustor
    prob.model.add_design_var('combustor.DL_hi', lower=2.5, upper=6.0)
    prob.model.add_design_var('combustor.dg_hi', lower=0.8, upper=1.2)
    prob.model.add_design_var('combustor.rd_DL', lower=0.25, upper=0.5)
    prob.model.add_design_var('combustor.theta', lower=np.deg2rad(7.0), upper=np.deg2rad(21.0))
    # prob.model.add_design_var('combustor.Mz3', lower=0.1, upper=0.3)
    prob.model.add_design_var('combustor.Aratio_p', lower=0.1, upper=0.8)
    prob.model.add_design_var('combustor.Aratio_d', lower=0.1, upper=0.8)
    prob.model.add_design_var('combustor.rm', lower=0.01, upper=0.03)
    prob.model.add_design_var('combustor.far', lower=0.016, upper=0.040)
    prob.model.add_design_var('combustor.he', lower=0.01, upper=0.05)

    # prob.model.add_constraint('combustor.PLF_liner', lower=5.0, upper=5.0)
    prob.model.add_constraint('combustor.lr_inner_3', lower=0.01, upper=0.1)
    prob.model.add_constraint('combustor.lr_outer_3', lower=0.01, upper=0.1)
    prob.model.add_constraint('combustor.To4', lower=900., upper=1450.)


    # Turbine

    # Design Variables
    for i8 in range(1, numStagesTurbine + 1):
        prob.model.add_design_var('turbine.T{0}.Re_n'.format(i8), lower=0.1E5, upper=3.2E5)
        prob.model.add_design_var('turbine.T{0}.Re_r'.format(i8), lower=0.1E5, upper=3.2E5)
        prob.model.add_design_var('turbine.T{0}.DR'.format(i8), lower=0.1, upper=1.0)
        prob.model.add_design_var('turbine.T{0}.workpercent'.format(i8), lower=0.1, upper=1.0)
        # prob.model.add_constraint('turbine.T{0}.M2_rel'.format(i8), lower=0.1, upper=1)
        engine.connect('turbine.T{0}.wdot_rotor'.format(i8), 'turbine.TurbinePerformance.wdot{0}'.format(i8))

    for i9 in range(2, numStagesTurbine + 1):
        prob.model.add_design_var('turbine.T{0}.M2'.format(i9), lower=0.1, upper=1)

    # prob.model.add_constraint('turbine.T{0}.alfa3'.format(numStagesTurbine), lower=0, upper=0)
    # TODO: Implement using actual wdot_hpc
    # prob.model.add_constraint('turbine.TurbinePerformance.wdotsum'.format(numStagesTurbine), lower=-wdot_hpc/0.99-wdot_hpc*0.1, upper=-wdot_hpc/0.99+wdot_hpc*0.1)

    engine.connect('combustor.Po4', 'turbine.TurbinePerformance.Po4')
    engine.connect('combustor.To4', 'turbine.TurbinePerformance.To4' )
    engine.connect('combustor.To4', 'turbine.To4')
    engine.connect('combustor.Po4', 'turbine.Po4')
    engine.connect('combustor.lM3', 'turbine.M4')
    engine.connect('compressor.N', 'turbine.N_rpms')

    prob.model.add_objective('compressor.n_overall', scaler=-1)  # minimization
    prob.setup()

    # Common Inputs
    engine.set_input_defaults('mdot', mdot)
    prob.set_val('mdot', mdot)

    # Diffuser
    prob.set_val("diffuser.Ma", M_flight)
    prob.set_val('P0_flight', P0_flight)
    prob.set_val('T0_flight', T0_flight)
    prob.set_val('diffuser.A_inlet', 0.4)
    prob.set_val('diffuser.Aratio', 0.95)

    # Compressor First Inputs
    prob.set_val('compressor.rm', 0.38)
    prob.set_val('compressor.cz', 200)
    prob.set_val('compressor.C1.fCoeff', 0.75)
    prob.set_val('compressor.C1.lCoeff', 0.33)
    prob.set_val('compressor.alp1', 0.57)
    prob.set_val('P0_flight', P0_flight)
    prob.set_val('T0_flight', T0_flight)
    prob.set_val('compressor.C1.T01', T0_flight)
    prob.set_val('compressor.numStages', numStages)


    for i5 in range(1, numStages + 1):
        prob.set_val('compressor.C{0}.lCoeff'.format(i5), 0.33)
        prob.set_val('compressor.C{0}.Df_r'.format(i5), 0.43)
        prob.set_val('compressor.C{0}.sigma_s'.format(i5), 0.56)
        prob.set_val('compressor.C{0}.chi3'.format(i5), 0.5)
        prob.set_val('compressor.C{0}.camber_s'.format(i5), 0.001)
        prob.set_val('compressor.C{0}.thickness'.format(i5), 0.1)
        prob.set_val('compressor.C{0}.b_s'.format(i5), 0.1)
        prob.set_val('compressor.C{0}.b_r'.format(i5), 0.1)

    # IGV First Inputs
    prob.set_val('compressor.IGV.P0in', P0_flight)
    prob.set_val('compressor.IGV.T0in', T0_flight)

    # prob.set_val('compressor.IGV.czin', 175)
    prob.set_val('compressor.IGV.b', 0.1)

    # Combustor Inputs

    prob.set_val('combustor.DL_hi', 3)
    prob.set_val('combustor.dg_hi', 0.8)
    prob.set_val('combustor.rd_DL', 0.4)
    prob.set_val('combustor.Mz3', 0.3)
    prob.set_val('combustor.theta', 0.209)
    prob.set_val('combustor.Aratio_p', 0.5)
    prob.set_val('combustor.Aratio_d', 0.5)
    prob.set_val('combustor.far', 0.020)
    prob.set_val('combustor.rm', 0.025)
    prob.set_val('combustor.he', 0.035)


    # Turbine Inputs


    prob.set_val('turbine.T1.alfa1', np.deg2rad(0))
    prob.set_val('turbine.wdot_hpc', wdot_hpc)
    prob.set_val('turbine.T1.workpercent', 0.55)
    prob.set_val('turbine.T2.workpercent', 0.45)


    for i9 in range(1, numStagesTurbine + 1):
        prob.set_val('turbine.T{0}.Re_n'.format(i9), 2.5E5)
        prob.set_val('turbine.T{0}.Re_r'.format(i9), 2.5E5)
        prob.set_val('turbine.T{0}.DR'.format(i9), 0.22)

    # prob.run_model()
    prob.run_driver()
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)

