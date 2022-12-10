import numpy as np
import openmdao.api as om

# Constraints

# Constant cz
# Constant rm
# Max M2rel = 0.75
# Max Df = 0.55
# Pressure loss * 2.3
# zero exit swirl
# Taper ratio 0.8
# Max thickness 10%


class CompressorFirstStage(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('gamma', default=1.4)
        self.options.declare('R', default=287.05)
        self.options.declare('OPR', default=11)
        self.options.declare('taper', default=0.8)

    def setup(self):

        # Picked variables
        self.add_input('rm', desc='mean radius')
        self.add_input('cz', desc='cz')
        self.add_input('fCoeff', desc='Flow Coefficient')
        self.add_input('lCoeff', desc='Loading Coefficient')
        self.add_input('alp1', desc='Alpha 1')
        self.add_input('Df_r', desc='Diffusion Factor Rotor')
        self.add_input('sigma_s', desc='Stator Solidity')
        self.add_input('camber_s', desc='Stator Camber')
        self.add_input('chi3', desc='Stator Camber')
        self.add_input('thickness', desc='thickness')
        self.add_input('b_s', desc='chord stator')
        self.add_input('b_r', desc='chord rotor')

        # Other Inputs
        self.add_input('T01', desc='Stagnation Temp In')
        self.add_input('P01', desc='Stagnation Pressure In')
        self.add_input('mdot', desc="Mass Flow Rate")
        self.add_input('numStages', desc='Number of Compressor Stages')


        self.add_output('P03', desc="P03")
        self.add_output('T03', desc="T03")
        self.add_output('n_st', desc="Stage efficiency")
        self.add_output('N', desc="RPM")
        self.add_output('r_h', desc="Hub radius")
        self.add_output('r_t', desc="Tip radius")
        self.add_output('Pr_st', desc="")
        self.add_output('Wdot', desc="Power")
        # self.add_output('camber_r', desc="Rotor camber")
        # self.add_output('chi1', desc="chi1")
        # self.add_output('chi_2r', desc="chi_2r")
        self.add_output('bet2', desc="Beta 2")
        self.add_output('sigma_r', desc="Rotor solidity")
        self.add_output('omega_r', desc="Rotor loss")
        self.add_output('Cp_r', desc="Cp_r")
        self.add_output('M1_rel', desc="M1_rel")
        # self.add_output('camber_s', desc="Stator camber")
        # self.add_output('chi_2s', desc="chi_2s")
        # self.add_output('chi3', desc="chi3")
        self.add_output('alp2', desc="Alpha 2")
        self.add_output('omega_s', desc="Stator loss")
        self.add_output('Df_s', desc="Diffusion factor stator")
        self.add_output('Cp_s', desc="Cp_s")
        self.add_output('M2', desc="Mach at 2")
        self.add_output('alp3', desc="alpha 3")
        self.add_output('delta_s', desc="delta s")
        self.add_output('sigma_c_over_rho_blade', desc="")
        self.add_output('sigma_bend', desc="")
        self.add_output('n_poly', desc="polytropic efficiency")
        self.add_output('T1', desc="")
        self.add_output('sqrtterm', desc="")
        self.add_output('n_blades_s', desc="")
        self.add_output('n_blades_r', desc="")
        self.add_output('h_o_b_s', desc="")
        self.add_output('h_o_b_r', desc="")
        self.add_output('AN2', desc="")
        self.add_output('P02', desc="")
        self.add_output('DR', desc="")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        rm = inputs['rm']
        cz = inputs['cz']
        fCoeff = inputs['fCoeff']
        lCoeff = inputs['lCoeff']
        alp1 = inputs['alp1']
        Df_r = inputs['Df_r']
        camber_s = inputs['camber_s']
        chi3 = inputs['chi3']
        T01 = inputs['T01']
        P01 = inputs['P01']
        mdot = inputs['mdot']
        numStages = inputs['numStages']
        sigma_s = inputs['sigma_s']
        thickness = inputs['thickness']
        b_s = inputs['b_s']
        b_r = inputs['b_r']


        gamma = self.options['gamma']
        R = self.options['R']
        OPR = self.options['OPR']
        taper = self.options['taper']

        cp = gamma * R / (gamma - 1)

        bet1 = np.arctan(np.tan(alp1)-1/fCoeff)
        # chi1 = bet1 + i_r
        bet2 = np.arctan(lCoeff/fCoeff+np.tan(bet1))
        alp2 = np.arctan(np.tan(bet2)+1/fCoeff)
        sigma_r = (np.cos(bet1)*(np.tan(bet2)-np.tan(bet1))/2)/(Df_r-1+np.cos(bet1)/np.cos(bet2))

        # delta_r = (bet2-chi1)/(np.sqrt(sigma_r)/0.25-1)
        # chi_2r = bet2 + delta_r
        # camber_r = chi1 - chi_2r
        # chi_2s = alp2 - i_s

        delta_s = 0.25*abs(camber_s)/np.sqrt(sigma_s)
        alp3 = delta_s+chi3
        # alp3 = alp1+np.deg2rad(1)
        # camber_s = chi_2s - chi3
        N = 30*cz/(fCoeff*np.pi*rm)

        Df_s = 1-np.cos(alp2)/np.cos(alp3)+np.cos(alp2)*(np.tan(alp2)-np.tan(alp3))/(sigma_s*2)
        omega_r = 2.3*((np.cos(bet1)/np.cos(bet2))**2*sigma_r*(0.012+0.0004*np.exp(7.5*Df_r))/np.cos(bet2))
        omega_s = 2.3*((np.cos(alp2)/np.cos(alp3))**2*sigma_s*(0.012+0.0004*np.exp(7.5*Df_s))/np.cos(alp3))

        T02 = T01 + (cz/fCoeff)**2*(lCoeff/cp)

        T1 = T01 - (cz/np.cos(alp1))**2/(2*cp)
        M1_rel = (cz/np.cos(bet1))/np.sqrt(gamma*R*T1)

        T2 = T02 - (cz/np.cos(alp2))**2/(2*cp)
        M2 = (cz/np.cos(alp2))/np.sqrt(gamma*R*T2)

        P1 = P01*(T1/T01)**(gamma/(gamma-1))
        M2_rel = (cz/np.cos(bet2))/np.sqrt(gamma*R*T2)

        P02 = P01*((((T02/T2)/(1+(gamma-1)*0.5*M2_rel**2))**(gamma/(gamma-1)))*(1-omega_r*(1-(1+(gamma-1)*0.5*M1_rel**2)**(gamma/(1-gamma))))*((1+(gamma-1)*0.5*M1_rel**2)/(T01/T1))**(gamma/(gamma-1)))
        P2 = P02*(T2/T02)**(gamma/(gamma-1))
        P03 = P02*(1-omega_s*(1-P2/P02))
        T3 = T02 - (cz/np.cos(alp3))**2/(2*cp)
        c3 = cz / np.cos(alp3)
        T03 = T3 + c3**2/(2*cp)

        P3 = P03*(T3/T02)**(gamma/(gamma-1))

        Cp_r = (P2/P1-1)/((1+(gamma-1)*0.5*M1_rel**2)**(gamma/(gamma-1))-1)
        Cp_s = (P3/P2-1)/((T02/T2)**(gamma/(gamma-1))-1)

        rho1 = P1/(R*T1)
        r_t = np.sqrt(rm**2+mdot/(2*np.pi*rho1*cz))
        r_h = np.sqrt(rm**2-mdot/(2*np.pi*rho1*cz))

        Wdot = mdot*cp*(T02-T01)

        # TODO: Implement stress constraints
        Omega = N*np.pi/30
        A_z = 2*np.pi*rm*(r_t-r_h)
        sigma_c_over_rho_blade = Omega**2*A_z*(1+taper)/(4*np.pi)
        U = N * np.pi * rm / 30
        delH0 = Wdot/mdot
        Utip = N*np.pi*r_t/30
        sigma_bend = P1*cz*abs(delH0)*(r_t/thickness)**2/(Utip*cp*T01*2*sigma_r)
        AN2 = A_z*N**2
        T03isen = (P03**0.285714*T01)/(P01**0.285714)
        n_poly = cp*(T03isen-T01)/delH0

        s_r = b_r/sigma_r
        s_s = b_s/sigma_s

        n_blades_r = 2*np.pi*rm/s_r
        n_blades_s = 2*np.pi*rm/s_s

        h_o_b_r = (r_t-r_h)/b_r
        h_o_b_s = (r_t-r_h)/b_s

        Pr_st = P03/P01
        n_st = (Pr_st**((gamma-1)/gamma)-1)/(T02/T01-1)

        DR = -0.5*(lCoeff-2+2*fCoeff*np.tan(alp2))

        chi1 = bet1 + 0
        delta_r = (bet2-chi1)/(np.sqrt(sigma_r)/0.25-1)
        chi_2r = bet2 + delta_r
        # print('Chi2r = {0} for P01 = {1}'.format(chi_2r*180/np.pi, P01))

        outputs['DR'] = DR
        outputs['P02'] = P02
        outputs['AN2'] = AN2
        outputs['h_o_b_r'] = h_o_b_r
        outputs['h_o_b_s'] = h_o_b_s
        outputs['n_blades_r'] = n_blades_r
        outputs['n_blades_s'] = n_blades_s
        outputs['P03'] = P03
        outputs['T03'] = T03
        outputs['n_st'] = n_st
        outputs['n_poly'] = n_poly
        outputs['r_h'] = r_h
        outputs['r_t'] = r_t
        outputs['sigma_bend'] = sigma_bend
        outputs['sigma_c_over_rho_blade'] = sigma_c_over_rho_blade
        outputs['N'] = N
        outputs['Wdot'] = Wdot
        # outputs['camber_r'] = camber_r
        # outputs['chi1'] = chi1
        # outputs['chi_2r'] = chi_2r
        outputs['bet2'] = bet2
        outputs['sigma_r'] = sigma_r
        outputs['omega_r'] = omega_r
        outputs['Cp_r'] = Cp_r
        outputs['M1_rel'] = M1_rel
        # outputs['camber_s'] = camber_s
        # outputs['chi_2s'] = chi_2s
        # outputs['chi3'] = chi3
        outputs['alp2'] = alp2
        outputs['omega_s'] = omega_s
        outputs['Df_s'] = Df_s
        outputs['Cp_s'] = Cp_s
        outputs['M2'] = M2
        outputs['alp3'] = alp3
        outputs['delta_s'] = delta_s
        outputs['Pr_st'] = Pr_st
        outputs['T1'] = T1
        outputs['sqrtterm'] = rm ** 2 - mdot / (2 * np.pi * rho1 * cz)


if __name__ == "__main__":
    prob = om.Problem()

    # Compressor First Stage

    prob.model.add_subsystem('C1', CompressorFirstStage(), promotes_inputs=['*'], promotes_outputs=['*'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.setup()

    # Compressor First Stage Inputs

    prob.set_val('rm', 0.53)
    prob.set_val('cz', 148)
    prob.set_val('fCoeff', 0.397)
    prob.set_val('lCoeff', 0.252)
    prob.set_val('alp1', np.deg2rad(54))
    prob.set_val('Df_r', 0.493)
    prob.set_val('sigma_s', 0.52)
    prob.set_val('T01', 273)
    prob.set_val('P01', 66800)
    prob.set_val('mdot', 64.5)
    prob.set_val('chi3', np.deg2rad(51.5))
    prob.set_val('camber_s', np.deg2rad(10))

    prob.run_model()
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)






