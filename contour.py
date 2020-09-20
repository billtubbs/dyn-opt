# Python functions to tune a PI controller using the
# contour method (É. Poulin and A. Pomerleau 1997)
# 
# There are two methods depending on the process type.
#
# Type 1. An asymptotically stable process:
# 
# Gp(s) = K * (1 - T0*s) * exp(-theta*s) / (1 + T1*s) * (1 + T2*s)
# 
# In this case, the tuning results in the output response of the
# process following a setpoint step (without filter) being
# approximately the fastest possible with an overshoot of
# approximately 8.5%.
# 
# Example:
# >>> K = 5; theta = 4; T0 = 2; T1 = 4; T2 = 3
# >>> Gp = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
# >>> PI = PIController()
# >>> PI.tune(Gp)
# PIController(Kc=0.08884158590979692, Ti=5.975, Tf=0)
#
#
# Type 2. A process with integration:
# 
# Gp(s) = K * exp(-theta*s) / (s * (1 + T*s))
# 
# In this case, the tuning sets the peak amplitude ratio
# of the closed-loop process to 4.4 dB by default.
#
# The controller is a PI with filter (PIF):
# 
# Gc(s) = Kc*(1 - Ti*s) / (Ti*s*(1 + Tf*s))
#
# Example:
# >>> K = 0.5; theta = 2; T = 3
# >>> Gp = ProcessWithIntegrator(K=K, theta=theta, T=T)
# >>> PI = PIController()
# >>> PI.tune(Gp)
# PIController(Kc=0.21815401860147718, Ti=23.420689994562874, Tf=0)


import math
import sympy
from sympy import Symbol
from sympy.solvers import nsolve
from pid import PIController


class StableProcess:
    """An asymptotically stable process with a transfer function
    of the following form:
    
    Gp(s) = K*(1 - T0*s)*exp(-theta*s) / (1 + T1*s)*(1 + T2*s)
    
    where T1 >= T2 >= 0, T0 >= 0, and s is the Laplace variable.
    
    Attributes:
        K (float): Process gain
        theta (float): Time delay (s)
        T0 (float): Time constant of process zero (s)
        T1 (float): Time constant #1 of process (s)
        T2 (float): Time constant #2 of process (s)
        omega_0 (float): Gain crossover frequency (rad/s)
    """
    # TODO: Generalize this to T0 < 0 and T0 > 0

    def __init__(self, K, theta, T0, T1, T2):
        self.K = K
        self.theta = theta
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.omega_0 = None

    def calculate_PI_parameters(self, Mr=0.25):
        """Finds the values of the PI controller parameters Kc and Ti such that
        the closed-loop system is stable and the frequency response of G(s)
        follows the 0.25 dB contour of the Nichols chart (a line of constant
        amplitude ratio of H(s)) without crossing it.
        
        With this method, the response of the resulting closed-loop system to a
        setpoint step is close to the fastest possible and has an overshoot of
        approximately 8.5%.
        """

        Ti = self.calculate_integration_constant(self.theta, self.T1, self.T2)
        self.omega_0 = self.calculate_crossover_frequency(self.theta, Ti, self.T0, self.T1, self.T2, Mr)
        Kc = self.calculate_gain(self.K, self.theta, self.omega_0, Ti, self.T0, self.T1, self.T2)
        return Kc, Ti

    @staticmethod
    def calculate_integration_constant(theta, T1, T2):
        if theta/T1 <= 2:
            return (1 + 0.175*theta/T1 + 0.3*(T2/T1)**2 + 0.2*T2/T1)*T1
        else:
            return (0.65 + 0.35*theta/T1 + 0.3*(T2/T1)**2 + 0.2*T2/T1)*T1

    @staticmethod
    def frequency_equation(omega, theta, Ti, T0, T1, T2, Mr):
        phi = math.acos(1 - 0.5*10**(-0.1*Mr)) - math.pi
        return (-phi - sympy.pi/2 + sympy.atan(omega*Ti) - sympy.atan(omega*T0) 
                - sympy.atan(omega*T1) - sympy.atan(omega*T2) - omega*theta)

    @staticmethod
    def calculate_crossover_frequency(theta, Ti, T0, T1, T2, Mr=0.25):
        omega_0 = Symbol('ω_0')
        expr = StableProcess.frequency_equation(omega_0, theta, Ti, T0, T1, T2, Mr)
        return nsolve(expr,omega_0,1)

    @staticmethod
    def calculate_gain(K, theta, omega_0, Ti, T0, T1, T2):
        return Ti/K*math.sqrt(
            ((T1*T2)**2*omega_0**6 + (T1**2 + T2**2)*omega_0**4 + omega_0**2) / 
            ((Ti*T0)**2*omega_0**4 + (Ti**2 + T0**2)*omega_0**2 + 1)
        )

    def __repr__(self):
        return f"StableProcess({self.K.__repr__()}, {self.theta.__repr__()}, " \
               f"{self.T0.__repr__()}, {self.T1.__repr__()}, {self.T2.__repr__()})"


class ProcessWithIntegrator:
    """An unstable process with a transfer function of the 
    following form:
    
    Gp(s) = K*exp(-theta*s) / (s*(1 + T*s))
    
    where T >= 0 and s is the Laplace variable.
    
    Attributes:
        K (float): Process gain
        theta (float): Time delay (s)
        T (float): Time constant of process
    """

    def __init__(self, K, theta, T):
        self.K = K
        self.theta = theta
        self.T = T
        self.Amax = None
        self.Phimax = None
        self.omega_max = None

    def calculate_PI_parameters(self, Mr=4.4):
        """Finds the values of the PI controller parameters Kc and Ti such that
        the closed-loop system is stable and the frequency response of G(s) at
        medium frequencies is closely aligned with the Nichols chart contour
        corresponding to the maximum peak resonance Mr (4.4 dB by default).
        """
        self.Amax = self.calculate_maximum_amplitude(Mr=Mr)
        self.Phimax = self.calculate_maximum_phase(self.Amax)
        Ti = self.calculate_integration_constant(self.theta, self.T, self.Amax, self.Phimax, Mr=4.4)
        self.omega_max = self.calculate_omega_max(self.theta, self.T, Ti)
        Kc = self.calculate_gain(self.K, self.T, Ti, self.Amax, self.omega_max)
        return Kc, Ti

    @staticmethod
    def calculate_maximum_amplitude(Mr=4.4):
        return 10**(0.05*Mr) / math.sqrt(10**(0.1*Mr) - 1)

    @staticmethod
    def calculate_maximum_phase(Amax):
        return -math.pi + math.acos(1/Amax)

    @staticmethod
    def calculate_integration_constant(theta, T, Amax, Phimax, Mr=4.4):
        return 16*(T + theta) / (2*Phimax + math.pi)**2

    @staticmethod
    def calculate_omega_max(theta, T, Ti):
        return 1 / math.sqrt(Ti*(T + theta))

    @staticmethod
    def calculate_gain(K, T, Ti, Amax, omega_max):
        return Ti*Amax / K*math.sqrt(
            ((T*omega_max)**2 + 1)*omega_max**4 / ((Ti*omega_max)**2 + 1)
        )

    def __repr__(self):
        return f"ProcessWithIntegrator({self.K.__repr__()}, " \
               f"{self.theta.__repr__()}, {self.T.__repr__()})"


def run_tests():
    """Test calculations"""

    # Example 14.11 from GEL-2005 course notes:
    K = 5; theta = 4; T0 = 2; T1 = 4; T2 = 3
    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    assert str(Gp1) == "StableProcess(5, 4, 2, 4, 3)"

    Ti = Gp1.calculate_integration_constant(theta, T1, T2)
    assert Ti == 5.975

    omega_0 = Gp1.calculate_crossover_frequency(theta, Ti, T0, T1, T2, Mr=0.25)
    assert (omega_0 - 0.0771077908075399) < 1e-10

    Kc = Gp1.calculate_gain(K, theta, omega_0, Ti, T0, T1, T2)
    assert (Kc - 0.08884158590979692) < 1e-10

    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    Kc2, Ti2 = Gp1.calculate_PI_parameters()
    assert Ti2 == Ti
    assert Kc2 == Kc

    PI1 = PIController()
    PI1.tune(Gp1)
    assert (PI1.Ti, PI1.Tf) == (Ti, 0)
    assert (PI1.Kc - Kc) < 1e-10

    # Example 14.12 from GEL-2005 course notes:
    K = 0.5; theta = 2; T = 3
    Gp2 = ProcessWithIntegrator(K=K, theta=theta, T=T)
    assert str(Gp2) == "ProcessWithIntegrator(0.5, 2, 3)"

    Amax = Gp2.calculate_maximum_amplitude(Mr=4.4)
    assert (Amax - 1.2530167991155927) < 1e-10

    Phimax = Gp2.calculate_maximum_phase(Amax)
    assert (Phimax - -2.4948882087176605) < 1e-10

    Ti = Gp2.calculate_integration_constant(theta, T, Amax, Phimax, Mr=4.4)
    assert (Ti - 23.420689994562874) < 1e-10

    omega_max = Gp2.calculate_omega_max(theta, T, Ti)
    assert (omega_max - 0.09240918819227639) < 1e-10

    Kc = Gp2.calculate_gain(K, T, Ti, Amax, omega_max)
    assert (Kc - 0.21815401860147718) < 1e-10

    Gp2 = ProcessWithIntegrator(K=K, theta=theta, T=T)
    Kc2, Ti2 = Gp2.calculate_PI_parameters()
    assert Ti2 == Ti
    assert Kc2 == Kc

    PI2 = PIController()
    PI2.tune(Gp2)
    assert (PI2.Kc, PI2.Ti, PI2.Tf) == (Kc, Ti, 0)

    # Exercise 14.7 from GEL-2005 course exercises:
    Gp = ProcessWithIntegrator(K=1, theta=0, T=1)
    PI = PIController().tune(Gp)
    assert (Gp.Amax - 1.253) < 0.001
    assert (Gp.Phimax + 2.495) < 0.001
    assert (Gp.omega_max - 0.462) < 0.001
    assert (PI.Kc - 0.579) < 0.001
    assert (PI.Ti - 4.684) < 0.001
    
    print('Tests completed.')


if __name__ == '__main__':
    run_tests()
