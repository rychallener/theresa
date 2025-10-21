class DummyConfig:
      """Dummy configuration object"""
      def __init__(self):
          self.star = self.Star()
          self.planet = self.Planet()

      class Star:
          """Star configuration parameters"""
          def __init__(self):
              self.m = 1.0        # Mass (solar masses)
              self.r = 1.0        # Radius (solar radii)
              self.prot = 25.0    # Rotation period (days)
              self.t = 5800.0     # Temperature (K)
              self.d = 10.0       # Distance (pc)
              self.z = 0.0        # Metallicity
              self.starspec = 'bbint'  # Blackbody spectrum

      class Planet:
          """Planet configuration parameters"""
          def __init__(self):
              self.m = 0.001      # Mass (solar masses) ~ 1 Jupiter mass
              self.r = 0.1        # Radius (solar radii) ~ 1 Jupiter radius
              self.p0 = 1.0       # Reference pressure (bar)
              self.porb = 3.5     # Orbital period (days)
              self.prot = 3.5     # Rotation period (days) - tidally locked
              self.Omega = 0.0    # Longitude of ascending node (deg)
              self.ecc = 0.0      # Eccentricity
              self.inc = 90.0     # Inclination (deg)
              self.w = 0.0        # Argument of periastron (deg)
              self.t0 = 0.0       # Time of transit (days)
              self.a = 0.05       # Semi-major axis (AU)
              self.b = 0.0        # Impact parameter

class DummyFit:
      """Dummy fit object for testing initsystem"""
      def __init__(self):
          self.cfg = DummyConfig()


  # Create the dummy fit object
def create_dummy_fit():
      """
      Create a dummy fit object for testing jaxoplanet initsystem.
      
      Returns
      -------
      fit : DummyFit
          A fit object with typical hot Jupiter parameters
      """
      return DummyFit()