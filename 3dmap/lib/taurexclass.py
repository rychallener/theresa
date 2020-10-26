import taurex
from taurex import chemistry
from taurex import model
import constants as c
import numpy as np

class ArrayGas(taurex.chemistry.Gas):
    """
    A Gas object for Tau-REx that allows the user to 
    pass in a custom abundance profile.
    """
    def __init__(self, molecule_name, abn):
        super().__init__('ArrayGas', molecule_name)
        self.abn = abn

    @property
    def mixProfile(self):
        return self.abn

class EmissionModel3D(taurex.model.EmissionModel):
    """
    A Tau-REx model that computes eclipse depth from a single
    grid element on the planet. Does NOT include visibility considerations.

    It would be faster, though more difficult, to overload path_integral.
    """
    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 ngauss=4,
                 latmin=-np.pi,
                 latmax=np.pi,
                 lonmin=0,
                 lonmax=2*np.pi
                 ):
        super().__init__(planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure,
                         ngauss)

        self.latmin = latmin * c.deg2rad
        self.latmax = latmax * c.deg2rad
        self.lonmin = lonmin * c.deg2rad
        self.lonmax = lonmax * c.deg2rad

    def compute_final_flux(self, f_total):
        # Catch for non-visible (far side) of the planet
        if ((self.lonmin >= np.pi / 2) or (self.lonmax <= -np.pi / 2)):
            return np.zeros(len(f_total))
        
        star_sed = self._star.spectralEmissionDensity

        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius

        #phimin   = np.max((self.lonmin,              -np.pi / 2))
        #phimax   = np.min((self.lonmax,               np.pi / 2))
        #thetamin = np.max((self.latmin + np.pi / 2.,  0))
        #thetamax = np.min((self.latmax + np.pi / 2.,  np.pi))
        phimin   = self.lonmin
        phimax   = self.lonmax
        thetamin = self.latmin + np.pi / 2.
        thetamax = self.latmax + np.pi / 2.

        # integral r**2 sin(theta) dtheta dphi
        planet_area = -1 * planet_radius**2  * (phimax - phimin) * \
            (np.cos(thetamax) - np.cos(thetamin))

        star_area = 2 * np.pi * star_radius **2

        cell_flux = (f_total / star_sed) * (planet_area / star_area)

        return cell_flux
