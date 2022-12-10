# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:16:41 2022
@author: The Compressors
"""

import numpy as np
import openmdao.api as om
from scipy import optimize

from CompressibleFlow import TR, PR


class TurbineStage(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_shaft', default=0.99)
        self.options.declare('psi_zn', default=-0.8)
        self.options.declare('psi_zr', default=0.8)
        self.options.declare('t_tr', default=0.5)
        self.options.declare('tmax', default=0.2)
        self.options.declare('gamma', default=1.28)
        self.options.declare('MW', default=28.7)

    def setup(self):
        # design variables
        # self.add_input('alfa2', desc="Rotor Inflow/Nozzle Outflow Angle")
        # self.add_input('alfa3', np.deg2rad(-29.1), units="rad", desc="Rotor Outflow Angle/Last Stage Exit Swirl Angle")
        # self.add_input('n_nblades', 0.0, desc="Number of Nozzle Blades")
        self.add_input('Re_n', desc="Throat Reynold's Number")
        self.add_input('Re_r', desc="Rotor Reynold's Number")
        self.add_input('DR', desc='Degree of Reaction')
        self.add_input('workpercent', desc='work percent')
        self.add_input('M2', desc='Mach at 2')
        self.add_input('rt', desc='rt')
        self.add_input('cz', desc='rt')

        # combustor exhaust conditions
        self.add_input('alfa1', desc="Combustor Exit Swirl/Nozzle Inflow Angle")
        self.add_input('mdot', desc="Combustor Product Flow Rate")
        self.add_input('To4', desc="Turbine Inlet Temperature")
        self.add_input('Po4', desc="Turbine Inlet Pressure")
        self.add_input('M4', desc="Combustor Exit Mach Number")
        # self.add_input('Vtheta', 0.0, units="m/s", desc="Swirl Velocity")
        self.add_input('wdot_hpc', desc="Required Power to Operate HPC at Design Point")
        self.add_input('N_rpms', desc="Shaft Speed")

        # overall 1st stage design output
        self.add_output('alfa2', desc="Nozzle Exit Angle")
        self.add_output('alfa3', desc="Rotor Exit Angle")
        self.add_output('phi', desc="Flow Coefficient")
        self.add_output('To_exit', desc="Exit Stagnation Temperature")
        self.add_output('Po_exit', desc="Exit Stagnation Pressure")
        self.add_output('n_stage', desc="Stage Efficiency")
        self.add_output('n_poly', desc="Polytropic Efficiency")
        self.add_output('rho_exit_inlet', desc="Exit-to-Inlet Density Ratio")

        # Nozzle design output
        self.add_output('sigma_n', desc="Nozzle Solidity")
        self.add_output('r_hub_nozzle', desc="Nozzle Hub Radius")
        self.add_output('r_tip_nozzle', desc="Nozzle Tip Radius")
        self.add_output('h_o_b_n', desc="Nozzle Blade Aspect Ratio")
        self.add_output('zeta_n', desc="Nozzle Energy Loss Coefficient")
        self.add_output('po2_po1', desc="Nozzle Pressure Ratio")
        self.add_output('n_bladesn', desc="Number of Nozzle Blades")
        self.add_output('n_bladesr', desc="Number of Nozzle Blades")

        # Rotor design output
        self.add_output('beta2', desc="Rotor Inlet Turn Angle")
        self.add_output('beta3', desc="Rotor Outlet Turn Angle")
        self.add_output('sigma_r', desc="Nozzle Solidity")

        self.add_output('h_br', desc="Nozzle Blade Aspect Ratio")
        self.add_output('zeta_r', desc="Rotor Energy Loss Coefficient")
        self.add_output('po3_po2', desc="Nozzle Pressure Ratio")
        self.add_output('M2_rel', desc="Relative Rotor Mach Number")
        self.add_output('AN2', desc="Rotor Material Limit Factor")
        self.add_output('centrifugal_stress', desc="Blade Centrifugal Stress")
        self.add_output('bending_stress', desc="Blade Bending Stress")
        self.add_output('wdot_rotor', desc="")
        self.add_output('chi3', desc="")
        self.add_output('sigma_c_rho_blade', desc="")
        self.add_output('h_o_b_r', desc="")
        self.add_output('psi', desc="")

        self.add_output('M_out', desc="Mach number out")

        # if the 'cs' method doesn't work, use 'fd' instead
        # uncomment when running the optimization
        self.declare_partials(of='*', wrt='*', method='cs', step=1e-9)

    def compute(self, inputs, outputs):
        # assumptions
        # * constant mass flow rate
        # * constant cz -> but density will change with constant mass flow,
        #   so axial area will change
        # * ideal CPG, < 2000 K
        # * adiabatic (not isentropic)
        # * no radial velocity
        # * no swirl velocity (inlet, exit)

        n_shaft = self.options['n_shaft']
        psi_zr = self.options['psi_zr']
        psi_zn = self.options['psi_zn']
        t_tr = self.options['t_tr']
        tmax = self.options['tmax']
        gamma = self.options['gamma']
        MW = self.options['MW']

        # Design Variables
        Re_n = inputs['Re_n']
        Re_r = inputs['Re_r']
        DR = inputs['DR']
        rt = inputs['rt']
        workpercent = inputs['workpercent']



        # Inputs
        To1 = inputs['To4']
        Po1 = inputs['Po4']
        M1 = inputs['M4']
        alfa1 = inputs['alfa1']
        wdot_hpc = inputs['wdot_hpc']
        mdot = inputs['mdot']
        N_rpms = inputs['N_rpms']
        cz = inputs['cz']


        # Flow Properties - using adiabatic relations
        R = 8314. / MW
        Cp = (gamma * R) / (gamma - 1)

        delta_ho = -wdot_hpc*workpercent/(n_shaft*mdot)

        U = np.sqrt(delta_ho/(2*DR-2))

        T1 = To1 / (1 + (gamma - 1) * 0.5 * M1 ** 2)
        P1 = Po1 / (1 + (gamma - 1) * 0.5 * M1 ** 2)
        mu = (0.000001458 * T1 ** (3 / 2)) / (T1 + 110.4)
        rho1 = P1 / (R * T1)

        c1 = M1 * np.sqrt(R * gamma * T1)

        phi = cz/U
        psi = 2*(DR-1)
        alfa2 = np.arctan(-2*(DR-1)/phi)
        c2 = cz / np.cos(alfa2)

        T2 = To1 - c2**2/(2*Cp)
        M2 = c2/np.sqrt(gamma*R*T2)

        beta3 = np.arctan((psi-2*DR)/(2*phi))

        rt1 = rt
        rm1 = np.sqrt((-2 * (mdot - 2 * cz * rho1 * rt1 ** 2 * np.pi)) / (cz * rho1)) / (2 * np.sqrt(np.pi))
        rh1 = np.sqrt(rm1 ** 2 - mdot / (2 * np.pi * rho1 * cz))
        h1 = rt1 - rh1

        sigma_zn_term1 = (gamma/(2*M2**2))/((1+(gamma-1)*0.5*M2**2)**(gamma/(gamma-1))-1)
        sigma_zn = (np.tan(alfa1) / np.tan(alfa2) - 1) * np.sin(2 * alfa2) * sigma_zn_term1 / psi_zn
        sigma_n = sigma_zn / np.cos(np.arctan((np.tan(alfa1) + np.tan(alfa2)) / 2))

        o_n = Re_n * (mu * R * T2 / c2) * ((To1 / T2) ** (gamma / (gamma - 1))) / Po1
        s_n = o_n / np.cos(alfa2)
        n_bladesn = np.round(2 * np.pi * rm1 / s_n)
        h_o_b_n = h1 / (s_n * sigma_n)
        b_zn_o_h = s_n * sigma_zn / h1

        Dhn = (2 * s_n * h1 * np.cos(alfa1)) / (s_n * np.cos(alfa2) + h1)
        Re_en = Re_n * Dhn / o_n
        zeta_n = (((1.04+0.06*((alfa1*180/np.pi+alfa2*180/np.pi)/100)**2)*(0.993+0.021*b_zn_o_h))-1)*(10**5/Re_en)**0.25


        Po2 = Po1 * ((1 - c2 ** 2 / (2 * Cp * To1 * (1 - zeta_n))) / (1 - c2 ** 2 / (2 * Cp * To1))) ** (
                    gamma / (gamma - 1))
        To2 = T2 + c2 ** 2 / (2 * Cp)
        P2 = Po2 * (T2 / To2) ** (gamma / (gamma - 1))
        rho2 = P2 / (R * T2)

        beta2 = np.arctan(np.tan(alfa2) - 1 / phi)
        alfa3 = np.arctan(np.tan(beta3) + 1 / phi)

        w2 = cz / np.cos(beta2)
        M2_rel = M2 * (w2 / c2)

        To3 = To1 + delta_ho / Cp

        c3 = cz / np.cos(alfa3)
        T3 = To3 - c3 ** 2 / (2 * Cp)
        w3 = cz / np.cos(beta3)

        M3_rel = w3 / np.sqrt(gamma * R * T3)

        sigma_zr_term1 = (gamma/(2*M3_rel**2))/((1+(gamma-1)*0.5*M3_rel**2)**(gamma/(gamma-1)))
        sigma_zr = (np.tan(beta2) / np.tan(beta3) - 1) * np.sin(2 * beta3) * sigma_zr_term1 / psi_zr
        sigma_r = sigma_zr / np.cos(np.arctan((np.tan(beta2) + np.tan(beta3)) / 2))

        o_r = Re_r * (mu * R * T3 / w3) * (To2 / T3) ** (gamma / (gamma - 1)) / Po2

        s_r = o_r / np.cos(beta3)
        h_o_b_r = h1 / (s_r * sigma_r)
        rt2 = rt1
        rm2 = np.sqrt((-2 * (mdot - 2 * cz * rho2 * rt2 ** 2 * np.pi)) / (cz * rho2)) / (2 * np.sqrt(np.pi))
        rh2 = np.sqrt(rm2 ** 2 - mdot / (2 * np.pi * rho2 * cz))
        h2 = rt2 - rh2

        b_zr_o_h = s_r * sigma_zr / h2

        Dhr = (2 * s_r * h1 * np.cos(beta3)) / (s_r * np.cos(beta3) + h1)
        Re_er = Re_r * Dhr / o_r
        zeta_r_term1 = (1.04+0.06*((beta2*180/np.pi+beta3*180/np.pi)/100)**2)
        zeta_r = ((zeta_r_term1*(0.975+0.075*b_zr_o_h))-1)*(10**5/Re_er)**0.25


        To2_rel = To1 - w2 ** 2 / (2 * Cp)
        # Po3_rel_o_Po2_rel = ((1 - w3 ** 2 / (2 * Cp * To2_rel * (1 - zeta_r))) / ( 1 - w3 ** 2 / (2 * Cp * To2_rel))) ** (gamma / (gamma - 1))
        P_3relterm1 = 1-w3**2/(2*Cp*To2_rel*(1-zeta_r))
        P_3relterm2 = 1-w3**2/(2*Cp*To2_rel)
        Po3_rel_o_Po2_rel = (P_3relterm1/P_3relterm2)**(gamma/(gamma-1))
        Po3_o_Po3_rel = (To3 / To2_rel) ** (gamma / (gamma - 1))
        Po2_rel_o_Po2 = (To2_rel / To1) ** (gamma / (gamma - 1))
        Po3 = Po3_o_Po3_rel * Po3_rel_o_Po2_rel * Po2_rel_o_Po2 * Po2
        P3 = Po3 * (T3 / To3) ** (gamma / (gamma - 1))

        rho3 = P3 / (R * T3)

        n_st = (1 - To3 / To1) / (1 - (Po3 / Po1) ** ((gamma - 1) / gamma))
        n_pt = (np.log(To1 / To3)) / (np.log(1 + (To1 / To3 - 1) / n_st))


        n_bladesr = np.round(2 * np.pi * rm2 / s_r)

        # Blade Stresses
        A_z = (2 * np.pi * rm2 * (rt2 - rh2))
        angular_vel = (N_rpms * np.pi) / 30
        centrifugal_stress = (angular_vel ** 2) * (A_z / (4 * np.pi)) * (1 + t_tr)


        bs_term1 = phi * ((1 + ((h2 / 2) / rm2)) ** -1)
        bs_term2 = np.abs(psi * (U ** 2)) / ((gamma / (gamma - 1)) * R * To1)
        bs_term3 = ((rm2 / h2) + (1 / 2)) / tmax
        bending_stress = bs_term1 * bs_term2 * (bs_term3 ** 2) * (1 / (2 * sigma_r)) * ((Po2 / PR(M2, gamma)))

        wdot_rotor = psi * U ** 2 * mdot

        delta = np.random.rand()*2
        chi3 = alfa3 - np.deg2rad(delta)
        outputs['chi3'] = chi3

        outputs['M_out'] = c3 / np.sqrt(gamma * R * T3)

        #        print(Re_r)

        # TODO fix this
        # n_rWhole = 1
        # n_nWhole = 1
        # if n_rblades[0].is_integer():
        #     n_rWhole = 1
        # else:
        #     n_rWhole = 0
        #
        # if n_nblades[0].is_integer():
        #     n_nWhole = 1
        # else:
        #     n_nWhole = 0

        # outputs['n_rWhole'] = n_rWhole
        # outputs['n_nWhole'] = n_nWhole

        outputs['sigma_c_rho_blade'] = centrifugal_stress / rho2
        outputs['h_o_b_r'] = h_o_b_r
        outputs['AN2'] = (N_rpms ** 2) * (3.14) * ((rt1 ** 2) - (rh1 ** 2))
        outputs['wdot_rotor'] = wdot_rotor
        outputs['phi'] = phi
        outputs['psi'] = psi
        outputs['alfa2'] = alfa2
        outputs['alfa3'] = alfa3
        outputs['beta3'] = beta3
        outputs['beta2'] = beta2
        outputs['M2_rel'] = M2_rel
        outputs['sigma_n'] = sigma_n
        outputs['r_hub_nozzle'] = rh1
        outputs['zeta_n'] = zeta_n
        outputs['To_exit'] = To3
        outputs['sigma_r'] = sigma_r
        outputs['n_bladesn'] = n_bladesn
        outputs['n_bladesr'] = n_bladesr
        outputs['h_o_b_n'] = h_o_b_n
        outputs['h_br'] = h1 / (s_r * sigma_r)
        outputs['zeta_r'] = zeta_r
        outputs['po2_po1'] = Po2 / Po1
        outputs['po3_po2'] = Po3 / Po2
        outputs['Po_exit'] = Po3
        outputs['rho_exit_inlet'] = rho3 / rho1
        outputs['n_stage'] = n_st
        outputs['n_poly'] = n_pt
        outputs['AN2'] = (N_rpms ** 2) * (3.14) * ((rt1 ** 2) - (rh1 ** 2))
        outputs['centrifugal_stress'] = centrifugal_stress
        outputs['bending_stress'] = bending_stress


if __name__ == "__main__":
    prob = om.Problem()
    prob.model.add_subsystem('turb_stage', TurbineStage(),
                             promotes_inputs=['*'],  # ['rm','Re_th','Re_r','DR','n_shaft','Mne1','psi_z','mdot', 'To4',
                             # 'Po4','MoleWeight','gamma','mu','M4','wdot_hpc','N_rpms'],
                             promotes_outputs=['alfa2', 'alfa3', 'n_nblades',
                                               'phi', 'psi', 'beta2', 'beta3',
                                               'n_stage', 'n_poly', 'centrifugal_stress',
                                               'bending_stress'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    # design variables
    # prob.model.add_design_var('alfa2', lower=np.deg2rad(60.), upper=np.deg2rad(80.))
    # prob.model.add_design_var('alfa3', lower=np.deg2rad(-30.), upper=np.deg2rad(0.))
    # prob.model.add_design_var('n_nblades', lower=25, upper=1000)
    prob.model.add_design_var('rm', lower=0.3, upper=3.0)
    prob.model.add_design_var('Re_th', lower=1.0E5, upper=3.2E5)
    prob.model.add_design_var('Re_r', lower=1.0E5, upper=3.2E5)
    prob.model.add_design_var('DR', lower=0.1, upper=1.0)

    # constraints / given values
    prob.model.add_constraint('alfa2', lower=np.deg2rad(60.), upper=np.deg2rad(80.))
    prob.model.add_constraint('alfa3', lower=np.deg2rad(-30.), upper=np.deg2rad(0.))
    prob.model.add_constraint('n_nblades', lower=25, upper=200)
    prob.model.add_constraint('n_shaft', lower=0.99, upper=0.99)
    prob.model.add_constraint('Mne1', lower=1.0, upper=1.0)
    prob.model.add_constraint('psi_z', lower=-0.8, upper=-0.8)
    # prob.model.add_constraint('DR', lower=0.1, upper=1.0)  # analyze only an impulse turbine
    prob.model.add_constraint('mdot', lower=48.5, upper=48.5)
    prob.model.add_constraint('To4', lower=1600, upper=1600)
    prob.model.add_constraint('Po4', lower=615, upper=615)
    prob.model.add_constraint('MoleWeight', lower=28.8, upper=28.8)
    prob.model.add_constraint('gamma', lower=1.29, upper=1.29)
    prob.model.add_constraint('mu', lower=5.60E-5, upper=5.60E-5)
    prob.model.add_constraint('M4', lower=0.44, upper=0.44)
    # prob.model.add_constraint('Vtheta', lower=0.0, upper=0.0)
    prob.model.add_constraint('wdot_hpc', lower=11.5, upper=11.5)
    prob.model.add_constraint('N_rpms', lower=9250., upper=9250.)
    # prob.model.add_constraint('n_nWhole', lower=1, upper=1)
    # prob.model.add_constraint('n_rWhole', lower=1, upper=1)

    prob.model.add_objective('n_stage', scaler=-1)
    prob.setup()

    # the actual values
    prob.set_val('alfa1', 0.0)
    prob.set_val('alfa3', np.deg2rad(-15.))
    prob.set_val('n_nblades', 30)
    prob.set_val('rm', 0.5)
    prob.set_val('Re_th', 1.2E5)
    prob.set_val('Re_r', 1.2E5)

    results = prob.run_driver()

    # for testing
    # 1. Max eff. case
    # =============================================================================
    # prob.set_val('alfa1', np.deg2rad(0.0))
    # prob.set_val('alfa2', np.deg2rad(62.33))
    # # prob.set_val('alfa3', np.deg2rad(-29.1))
    # prob.set_val('n_nblades', 41)
    # prob.set_val('rm', 0.3)
    # prob.set_val('Re_th', 1.88E5)
    # prob.set_val('n_shaft', 0.99)
    # prob.set_val('Mne1', 1.0)
    # prob.set_val('psi_z',-0.8)
    # prob.set_val('DR', 0.22)
    # prob.set_val('mdot', 48.5)
    # prob.set_val('To4', 1600)
    # prob.set_val('Po4', 615)
    # prob.set_val('MoleWeight', 28.8)
    # prob.set_val('gamma', 1.29)
    # prob.set_val('mu', 5.6E-5)
    # prob.set_val('M4', 0.44)
    # prob.set_val('Vtheta', 0.0)
    # prob.set_val('wdot_hpc', 11.5)
    # prob.set_val('N_rpms', 9250.)
    #
    # =============================================================================
    # 2. Low eff. case

    # prob.run_model()
    # for testing
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)


