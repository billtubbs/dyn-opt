# Python classes to represent continous time, single-input, 
# single-output (SISO) dynamical system models.


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

    def __repr__(self):
        return f"ProcessWithIntegrator({self.K.__repr__()}, " \
               f"{self.theta.__repr__()}, {self.T.__repr__()})"


def run_tests():

    # Example 14.11 from GEL-2005 course notes:
    K = 5; theta = 4; T0 = 2; T1 = 4; T2 = 3
    Gp1 = StableProcess(K=K, theta=theta, T0=T0, T1=T1, T2=T2)
    assert str(Gp1) == "StableProcess(5, 4, 2, 4, 3)"

    # Example 14.12 from GEL-2005 course notes:
    K = 0.5; theta = 2; T = 3
    Gp2 = ProcessWithIntegrator(K=K, theta=theta, T=T)
    assert str(Gp2) == "ProcessWithIntegrator(0.5, 2, 3)"

    print('Tests completed.')


if __name__ == '__main__':
    run_tests()
