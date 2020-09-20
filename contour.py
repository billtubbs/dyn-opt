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
from siso import StableProcess, ProcessWithIntegrator


def integration_constant_StableProcess(theta, T1, T2):
    if theta/T1 <= 2:
        return (1 + 0.175*theta/T1 + 0.3*(T2/T1)**2 + 0.2*T2/T1)*T1
    else:
        return (0.65 + 0.35*theta/T1 + 0.3*(T2/T1)**2 + 0.2*T2/T1)*T1


def frequency_equation_StableProcess(omega, theta, Ti, T0, T1, T2, Mr):
    phi = math.acos(1 - 0.5*10**(-0.1*Mr)) - math.pi
    return (-phi - sympy.pi/2 + sympy.atan(omega*Ti) - sympy.atan(omega*T0) 
            - sympy.atan(omega*T1) - sympy.atan(omega*T2) - omega*theta)


def crossover_frequency_StableProcess(theta, Ti, T0, T1, T2, Mr=0.25):
    omega_0 = Symbol('ω_0')
    expr = frequency_equation_StableProcess(omega_0, theta, Ti, T0, T1, T2, Mr)
    return nsolve(expr,omega_0,1)


def PI_gain_StableProcess(K, theta, omega_0, Ti, T0, T1, T2):
    return Ti/K*math.sqrt(
        ((T1*T2)**2*omega_0**6 + (T1**2 + T2**2)*omega_0**4 + omega_0**2) / 
        ((Ti*T0)**2*omega_0**4 + (Ti**2 + T0**2)*omega_0**2 + 1)
        )


def maximum_amplitude(Mr=4.4):
    return 10**(0.05*Mr) / math.sqrt(10**(0.1*Mr) - 1)


def maximum_phase(Amax):
    return -math.pi + math.acos(1/Amax)


def integration_constant_ProcessWithIntegrator(theta, T, Amax, Phimax, Mr=4.4):
    return 16*(T + theta) / (2*Phimax + math.pi)**2


def omega_max_ProcessWithIntegrator(theta, T, Ti):
    return 1 / math.sqrt(Ti*(T + theta))


def PI_gain_ProcessWithIntegrator(K, T, Ti, Amax, omega_max):
    return Ti*Amax / K*math.sqrt(
        ((T*omega_max)**2 + 1)*omega_max**4 / ((Ti*omega_max)**2 + 1)
    )


def tune_PI(Gp, Mr=None):
    """Finds the values of the PI controller parameters Kc and Ti such that
    the closed-loop system is stable and the frequency response of G(s)
    follows a contour at medium frequencies.

    If Gp is an instance of StableProcess:
    
        The PI is tuned such that the frequency response of G(s) follows
        the 0.25 dB contour of the Nichols chart (a line of constant
        amplitude ratio of H(s)) without crossing it.
    
    If Gp is an instance of ProcessWithIntegrator:

        The PI is tuned such that the frequency response of G(s) is
        closely aligned with the Nichols chart contour corresponding to
        the maximum peak resonance Mr (4.4 dB by default). The response
        of the resulting closed-loop system to a setpoint step is close
        to the fastest possible and has an overshoot of approximately 8.5%.
    """

    if isinstance(Gp, StableProcess):
        if Mr is None:
            Mr = 0.25
        Ti = integration_constant_StableProcess(Gp.theta, Gp.T1, Gp.T2)
        omega_0 = crossover_frequency_StableProcess(Gp.theta, Ti, Gp.T0, Gp.T1, Gp.T2, Mr=Mr)
        Kc = PI_gain_StableProcess(Gp.K, Gp.theta, omega_0, Ti, Gp.T0, Gp.T1, Gp.T2)

        return Kc, Ti

    elif isinstance(Gp, ProcessWithIntegrator):
        if Mr is None:
            Mr = 4.4
        Amax = maximum_amplitude(Mr=Mr)
        Phimax = maximum_phase(Amax)
        Ti = integration_constant_ProcessWithIntegrator(Gp.theta, Gp.T, Amax, Phimax, Mr=Mr)
        omega_max = omega_max_ProcessWithIntegrator(Gp.theta, Gp.T, Ti)
        Kc = PI_gain_ProcessWithIntegrator(Gp.K, Gp.T, Ti, Amax, omega_max)

        return Kc, Ti


def tune(Gc, Gp, Mr=None):
    if isinstance(Gc, PIController):
        Gc.Kc, Gc.Ti = tune_PI(Gp, Mr=Mr)
    else:
        raise NotImplementedError()
    return True


def run_tests():
    """Unit tests"""

    # Example 14.11 from GEL-2005 course notes:
    K = 5; theta = 4; T0 = 2; T1 = 4; T2 = 3
    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    Ti = integration_constant_StableProcess(theta, T1, T2)
    assert Ti == 5.975

    omega_0 = crossover_frequency_StableProcess(theta, Ti, T0, T1, T2, Mr=0.25)
    assert (omega_0 - 0.0771077908075399) < 1e-10

    Kc = PI_gain_StableProcess(K, theta, omega_0, Ti, T0, T1, T2)
    assert (Kc - 0.08884158590979692) < 1e-10

    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    Kc2, Ti2 = tune_PI(Gp1)
    assert Ti2 == Ti
    assert Kc2 == Kc

    # Example 14.12 from GEL-2005 course notes:
    K = 0.5; theta = 2; T = 3
    Gp2 = ProcessWithIntegrator(K=K, theta=theta, T=T)

    Amax = maximum_amplitude(Mr=4.4)
    assert (Amax - 1.2530167991155927) < 1e-10

    Phimax = maximum_phase(Amax)
    assert (Phimax - -2.4948882087176605) < 1e-10

    Ti = integration_constant_ProcessWithIntegrator(theta, T, Amax, Phimax, Mr=4.4)
    assert (Ti - 23.420689994562874) < 1e-10

    omega_max = omega_max_ProcessWithIntegrator(theta, T, Ti)
    assert (omega_max - 0.09240918819227639) < 1e-10

    Kc = PI_gain_ProcessWithIntegrator(K, T, Ti, Amax, omega_max)
    assert (Kc - 0.21815401860147718) < 1e-10

    Kc2, Ti2 = tune_PI(Gp2)
    assert Ti2 == Ti
    assert Kc2 == Kc

    # Exercise 14.7 from GEL-2005 course exercises:
    Gp = ProcessWithIntegrator(K=1, theta=0, T=1)
    Mr = 4.4
    Amax = maximum_amplitude(Mr)
    Phimax = maximum_phase(Amax)
    Ti = integration_constant_ProcessWithIntegrator(Gp.theta, Gp.T, Amax, Phimax, Mr)
    omega_max = omega_max_ProcessWithIntegrator(Gp.theta, Gp.T, Ti)
    assert (Amax - 1.253) < 0.001
    assert (Phimax + 2.495) < 0.001
    assert (omega_max - 0.462) < 0.001

    print('Tests completed.')


if __name__ == '__main__':
    run_tests()
