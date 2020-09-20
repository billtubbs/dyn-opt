from pid import PIController
from siso import StableProcess, ProcessWithIntegrator
from pid_tuning import tune


def run_tests():
    """Test calculations"""

    # Example 14.11 from GEL-2005 course notes:
    K = 5; theta = 4; T0 = 2; T1 = 4; T2 = 3
    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    PI1 = PIController()
    tune(PI1, Gp1, method='contour')
    assert (PI1.Ti, PI1.Tf) == (5.975, 0)
    assert (PI1.Kc - 0.08884158590979692) < 1e-10

    # Example 14.12 from GEL-2005 course notes:
    K = 0.5; theta = 2; T = 3
    Gp2 = ProcessWithIntegrator(K=K, theta=theta, T=T)
    PI2 = PIController()
    tune(PI2, Gp2, method='contour')
    assert (PI2.Kc, PI2.Ti, PI2.Tf) == (0.21815401860147718, 23.420689994562874, 0)

    # Exercise 14.7 from GEL-2005 course exercises:
    Gp = ProcessWithIntegrator(K=1, theta=0, T=1)
    PI = PIController()
    tune(PI, Gp, method='contour')
    assert (PI.Kc - 0.579) < 0.001
    assert (PI.Ti - 4.684) < 0.001
    
    print('Tests completed.')


if __name__ == '__main__':
    run_tests()
