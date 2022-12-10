import numpy as np
import openmdao.api as om


class InletGuideVane(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('gamma', default=1.4)
        self.options.declare('R', default=287.05)
        self.options.declare('Zweiffel', default=0.8)

    def setup(self):
        self.add_input('czin', desc='cz from diffuser')
        self.add_input('P0in', desc='stagnation pressure in')
        self.add_input('T0in', desc='stagnation temp in')
        self.add_input('cz', desc='axial speed into compressor')
        self.add_input('alp1', desc="flow angle into compressor")
        self.add_input('rm', desc="mean radius")
        self.add_input('mdot', desc='Mass Flow Rate')
        self.add_input('b', desc='chord')

        self.add_output('P0out', desc="stagnation pressure out")
        self.add_output('T0out', desc="stagnation temp out")
        self.add_output('fracPLoss', desc="")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        czin = inputs['czin']
        P0in = inputs['P0in']
        T0in = inputs['T0in']
        rm = inputs['rm']
        mdot = inputs['mdot']
        alp1 = inputs['alp1']
        b = inputs['b']
        cz = inputs['cz']



        R = self.options['R']
        gamma = self.options['gamma']
        Zweiffel = self.options['Zweiffel']

        outputs['T0out'] = T0in
        cp = gamma * R / (gamma - 1)

        Tin = T0in - czin**2/(2*cp)
        mu = (0.000001458*Tin**(3/2))/(Tin+110.4)
        a = np.sqrt(gamma*R*Tin)
        M_in = czin/a
        Pin = P0in/((1+(gamma-1)*0.5*M_in**2)**(gamma/(gamma-1)))
        Rhoin = Pin / (R*Tin)

        r_t = np.sqrt(rm**2+mdot/(2*np.pi*Rhoin*czin))
        r_h = np.sqrt(rm**2-mdot/(2*np.pi*Rhoin*czin))

        h = r_t-r_h

        sigma_IGV = Zweiffel * 2*np.cos(alp1)*np.sin(alp1)  # Flow turn from 0 deg to alp1 deg
        # sigma_zIGV = sigma_IGV bc no swirl incoming

        s = b/sigma_IGV

        C_nozzle = 0.993 + 0.021*(b/h)
        zetastar = 1.04+0.06*(np.deg2rad(alp1)/100)**2
        Dh = 2*s*h*np.cos(alp1)/(s*np.cos(alp1)+h)
        Ree = Rhoin * czin * Dh / mu
        zeta = (zetastar*C_nozzle-1)*(10**5/Ree)**0.25

        fracPLoss = 1-((1-cz**2/(2*cp*T0in*(1-zeta)))/(1-cz**2/(2*cp*T0in)))**(gamma/(gamma-1))
        P0out = -P0in*(fracPLoss-1)
        outputs['P0out'] = P0out
        outputs['fracPLoss'] = fracPLoss


if __name__ == "__main__":
    prob = om.Problem()

    # Compressor First Stage

    prob.model.add_subsystem('IGV', InletGuideVane(), promotes_inputs=['*'], promotes_outputs=['*'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-2

    prob.model.add_design_var('czin', lower=1, upper=200)
    # prob.model.add_design_var('rm', lower=0.1, upper=0.7)
    prob.model.add_design_var('b', lower=0.1, upper=0.7)

    prob.model.add_constraint('cz', lower=199.21, upper=199.21)
    prob.model.add_constraint('alp1', lower=0.57381526, upper=0.57381526)

    prob.model.add_objective('fracPLoss')

    prob.setup()


    # Compressor First Stage Inputs

    prob.set_val('rm', 0.38)
    prob.set_val('cz', 199.21)
    prob.set_val('alp1', np.deg2rad(54))
    prob.set_val('T0in', 288.15)
    prob.set_val('P0in', 80000)
    prob.set_val('mdot', 46)
    prob.set_val('czin', 250)
    prob.set_val('b', 0.1)

    prob.run_driver()
    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)









