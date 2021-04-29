class Configuration:
    """
    A class to hold parameters from a configuration file.
    """
    def __init__(self):
        self.planet = Planet()
        self.star   = Star()
        self.twod   = TwoD()
        self.threed = ThreeD()

class Planet:
    """
    A class to hold planet parameters.
    """
    pass

class Star:
    """
    A class to hold star parameters.
    """
    pass

class TwoD:
    """
    A class to hold 2D configuration options.
    """
    pass

class ThreeD:
    """
    A class to hold 3D configuration options.
    """
    pass
