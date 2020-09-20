# Python classes to represent proportional integral
# derivative (PID) controllers.


class ProportionalController:
    """Proportional controller.
    """

    def __init__(self, Kc=None):
        self.Kc = Kc

    def tune(self, Gp, *args, **kwargs):
        self.Kc = Gp.calculate_proportional(*args, **kwargs)
        return self

    def __repr__(self):
        return f"ProportionalController(Kc={self.Kc.__repr__()})"


class PDFController:
    """Proportional-derivative-filter (PDF) controller 
    or 'lead-lag compensator'
    """

    def __init__(self, Kc=None, Td=None, Tf=None):
        self.Kc = Kc
        self.Td = Td
        self.Tf = Tf

    def tune(self, Gp, *args, **kwargs):
        self.Kc, self.Td, self.Tf = Gp.calculate_PDF_parameters(*args, **kwargs)
        return self

    def __repr__(self):
        return f"PDFController(Kc={self.Kc.__repr__()}, Td={self.Td.__repr__()}, " \
               f"Tf={self.Tf.__repr__()})"


class PIController:
    """Proportional-integral (PI) controller
    """

    def __init__(self, Kc=None, Ti=None, Tf=0):
        self.Kc = Kc
        self.Ti = Ti
        self.Tf = Tf

    def tune(self, Gp, *args, **kwargs):
        self.Kc, self.Ti = Gp.calculate_PI_parameters(*args, **kwargs)
        return self

    def __repr__(self):
        return f"PIController(Kc={self.Kc.__repr__()}, Ti={self.Ti.__repr__()}, " \
               f"Tf={self.Tf.__repr__()})"


class PIDController:
    """Proportional-integral-derivative (PI) controller
    """

    def __init__(self, Kc=None, Ti=None, Td=None, Tf=0):
        self.Kc = Kc
        self.Ti = Ti
        self.Td = Td
        self.Tf = Tf

    def tune(self, Gp, *args, **kwargs):
        self.Kc, self.Ti = Gp.calculate_PID_parameters(*args, **kwargs)
        return self

    def __repr__(self):
        return f"PIDController(Kc={self.Kc.__repr__()}, Ti={self.Ti.__repr__()}, " \
               f"Td={self.Td.__repr__()}, Tf={self.Tf.__repr__()})"
