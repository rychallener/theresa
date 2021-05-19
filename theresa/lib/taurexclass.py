import taurex
from taurex import chemistry
from taurex import model
from taurex.util.emission import black_body
from taurex.constants import PI
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
        planet_area = np.pi * planet_radius ** 2

        star_area = np.pi * star_radius ** 2

        cell_flux = (f_total / star_sed) * (planet_area / star_area)

        return cell_flux

    def path_integral(self, wngrid, return_contrib):
        '''
        Overload the base emission path_integral() method to 
        return the actual cumulative tau array rather than
        an array of the change in transmittance. This gives
        more flexibility for further calculations.
        '''
        dz = np.gradient(self.altitudeProfile)

        density = self.densityProfile

        wngrid_size = wngrid.shape[0]

        total_layers = self.nLayers

        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))

        layer_tau = np.zeros(shape=(1, wngrid_size))

        dtau = np.zeros(shape=(1, wngrid_size))

        # Do surface first
        # for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(self, 0, total_layers, 0, 0,
                               density, surface_tau, path_length=dz)
        self.debug('density = %s', density[0])
        self.debug('surface_tau = %s', surface_tau)

        BB = black_body(wngrid, temperature[0])/PI

        _mu = 1.0/self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        I = BB * (np.exp(-surface_tau*_mu))

        self.debug('I1_pre %s', I)
        # Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(self, layer+1, total_layers,
                                   0, 0, density, layer_tau, path_length=dz)
                contrib.contribute(self, layer, layer+1, 0,
                                   0, density, dtau, path_length=dz)

            #_tau = np.exp(-layer_tau) - np.exp(-dtau)
            _tau = layer_tau + dtau

            tau[layer] += _tau[0]
            # for contrib in self.contribution_list:

            self.debug('Layer_tau[%s]=%s', layer, layer_tau)

            dtau += layer_tau
            self.debug('dtau[%s]=%s', layer, dtau)
            BB = black_body(wngrid, temperature[layer])/PI
            self.debug('BB[%s]=%s,%s', layer, temperature[layer], BB)
            I += BB * (np.exp(-layer_tau*_mu) - np.exp(-dtau*_mu))

        self.debug('I: %s', I)

        flux_total = 2.0 * np.pi * sum(I * (_w / _mu))
        self.debug('flux_total %s', flux_total)

        return self.compute_final_flux(flux_total).flatten(), tau
                         
