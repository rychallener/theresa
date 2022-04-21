import taurex
from taurex import chemistry
from taurex import model
from taurex import contributions
from taurex import constants
from taurex.util.emission import black_body
from taurex.constants import PI, PLANCK, SPDLIGT, KBOLTZ
import constants as c
import numpy as np
from numba import njit

hc_kb = PLANCK * SPDLIGT / KBOLTZ 

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
    grid element on the planet. Does NOT include visibility or
    grid cell size considerations.

    path_integral() has been modified to return the true optical
    depth array. Also, the emission spectrum is calculated from
    the top of the atmoshere down, and calculation ends when an
    optical depth limit is reached. The bottom of the returned
    optical depth array matches the deepest layer that is computed.
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
                 taulimit=5,
                 ngauss=4,
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
        
        self.taulimit = taulimit

    def compute_final_flux(self, f_total):
        
        star_sed = self._star.spectralEmissionDensity

        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius

        planet_area = planet_radius ** 2

        star_area = star_radius ** 2

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
        # Loop downwards
        for layer in range(total_layers - 1, -1, -1):
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

            self.debug('Layer_tau[%s]=%s', layer, layer_tau)

            dtau += layer_tau
            self.debug('dtau[%s]=%s', layer, dtau)
            BB = black_body(wngrid, temperature[layer])/PI
            self.debug('BB[%s]=%s,%s', layer, temperature[layer], BB)
            I += BB * (np.exp(-layer_tau*_mu) - np.exp(-dtau*_mu))
            self.debug('I_tot[%s]=%s', layer, I)

            if np.all(tau[layer] > self.taulimit):
                tau[:layer] += _tau[0]
                break

        self.debug('I: %s', I)

        flux_total = 2.0 * np.pi * sum(I * (_w / _mu))
        self.debug('flux_total %s', flux_total)

        return self.compute_final_flux(flux_total).flatten(), tau
                         
class HMinusContribution(taurex.contributions.Contribution):
    '''
    A class to compute the H- continuum opacity contribution from both
    free-free and bound-free interactions.
    '''
    
    def __init__(self):
        super().__init__('H-')

        self.fftbl1 = \
            np.array([[0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0],
                      [2483.3460, 285.8270, -2054.2910,
                       2827.7760, -1341.5370, 208.9520],
                      [-3449.8890, -1158.3820, 8746.5230,
                       -11485.6320, 5303.6090, -812.9390],
                      [2200.0400, 2427.7190, -13651.1050,
                       16755.5240, -7510.4940, 1132.7380],
                      [-696.2710, -1841.4000, 8624.9700,
                       -10051.5300, 4400.0670, -655.0200],
                      [88.2830, 444.5170, -1863.8640,
                       2095.2880, -901.7880, 132.9850]])

        self.fftbl2 = \
            np.array([[518.1021, -734.8666, 1021.1775,
                       -479.0721, 93.1373, -6.4285],
                      [473.2636, 1443.4137, -1977.3395,
                       922.3575, -178.9275, 12.3600],
                      [-482.2089, -737.1616, 1096.8827,
                       -521.1341, 101.7963, -7.0571],
                      [115.5291, 169.6374, -245.6490,
                       114.2430, -21.9972, 1.5097],
                      [0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0]])

        self.bftbl = \
            np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])

    def prepare_each(self, model, wngrid):
        self._nlayers = model.nLayers
        self._ngrid   = wngrid.shape[0]

        wlgrid = 10000. / wngrid

        chemistry = model.chemistry
        pressure = model.pressureProfile
        mix_profile = chemistry.get_gas_mix_profile('H-')

        sigma_hminus = np.zeros((self._nlayers, self._ngrid))

        for ilayer, temperature in enumerate(model.temperatureProfile):
            kff = freefree( self.fftbl1, self.fftbl2, temperature, wlgrid)
            kbf = boundfree(self.bftbl,               temperature, wlgrid)
            weight = mix_profile[ilayer]**2 * pressure[ilayer]
            sigma_hminus[ilayer] += (kff + kbf) * weight

        self.sigma_xsec = sigma_hminus

        yield self._name, sigma_hminus


@njit
def boundfree(bftbl, T, wl_um):
    '''
    Calculate bound-free (photo-detachment) opacity from John 1988.
    Converted to m^2/Pa from cm^2/Ba (factor of 0.001).
    Units are absorption (cross-section) per unit electron pressure
    per H- atom.
    '''
    # I'm pretty sure this is what alpha should be.
    # John 1988 might be off by a few orders of magnitude
    # Factor of 1e6 is m -> um to keep exponents unitless
    alpha   = hc_kb
    lambda0 = 1.6419 # um

    f   = np.zeros(len(wl_um))
    sig = np.zeros(len(wl_um))
    kbf = np.zeros(len(wl_um))

    idx = np.where(wl_um < lambda0)

    for n in range(len(bftbl)):
        # n/2 instead of (n-1)/2 because n starts at 0
        f[idx] += bftbl[n] * (1 / wl_um[idx] - 1 / lambda0)**(n/2.)

    sig[idx] = 1e-18 * wl_um[idx]**3 * \
        (1 / wl_um[idx] - 1 / lambda0)**(3./2.) * f[idx]

    # 7.5e-1 -> 7.5e-4 for cm^2/Ba -> m^2/Pa
    kbf[idx] = 7.5e-4 * T**(-5./2.) * np.exp(alpha / lambda0 / T) * \
        (1 - np.exp(-1 * alpha / wl_um[idx] / T)) * sig[idx]

    return kbf

@njit
def freefree(fftbl1, fftbl2, T, wl_um):
    '''
    Calculate free-free H- opacity from John 1988.
    Converted to m^2/Pa from cm^2/Ba (factor of 0.001).
    '''
    kff = np.zeros(len(wl_um))

    idx1 = np.where(wl_um >= 0.3645)
    idx2 = np.where((wl_um > 0.1823) & (wl_um < 0.3645))

    for n in range(fftbl1.shape[0]):
        # 1e-29 -> 1e-32 for cm^2/Ba -> m^2/Pa
        # n+2 in exponent because n starts at 0
        kff[idx1] += 1e-32 * (5040. / T)**((n+2)/2.) * \
            (fftbl1[n,0] * wl_um[idx1]**2 +
             fftbl1[n,1]                  +
             fftbl1[n,2] / wl_um[idx1]    +
             fftbl1[n,3] / wl_um[idx1]**2 +
             fftbl1[n,4] / wl_um[idx1]**3 +
             fftbl1[n,5] / wl_um[idx1]**4)
        kff[idx2] += 1e-32 * (5040. / T)**((n+2)/2.) * \
            (fftbl1[n,0] * wl_um[idx2]**2 +
             fftbl1[n,1]                  +
             fftbl1[n,2] / wl_um[idx2]    +
             fftbl1[n,3] / wl_um[idx2]**2 +
             fftbl1[n,4] / wl_um[idx2]**3 +
             fftbl1[n,5] / wl_um[idx2]**4)

    return kff

class LeeMieVaryMixContribution(taurex.contributions.LeeMieContribution):
    
    def __init__(self, molname=None, lee_mie_radius=0.01, lee_mie_q=40,
                 lee_mie_mix_ratio=1e-10, lee_mie_bottomP=-1,
                 lee_mie_topP=-1):
        super().__init__(lee_mie_radius=lee_mie_radius,
                         lee_mie_q=lee_mie_q,
                         lee_mie_mix_ratio=lee_mie_mix_ratio,
                         lee_mie_bottomP=lee_mie_bottomP,
                         lee_mie_topP=lee_mie_topP)
        if molname is None:
            self._name = 'Mie'
        else:
            self._name = 'Mie ' + molname

    def prepare_each(self, model, wngrid):
        """
        Computes and weights the mie opacity for
        the pressure regions given
        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model
        wngrid: :obj:`array`
            Wavenumber grid
        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Lee`` and the weighted mie opacity.
        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_profile = model.pressureProfile

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = pressure_profile[0]

        top_pressure = self.mieTopPressure
        if top_pressure < 0:
            top_pressure = pressure_profile[-1]

        cloud_idx = np.where((pressure_profile <= bottom_pressure) & \
            (pressure_profile >= top_pressure))[0]

        wltmp = 10000/wngrid

        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))
        
        for idx in cloud_idx:
            if ((self.mieRadius[idx] != 0.0) and
                (self.mieMixing[idx] != 0.0) and
                (self.mieQ[idx]      != 0.0)):
                a = self.mieRadius[idx]

                x = 2.0 * np.pi * a / wltmp
                self.debug('wngrid %s', wngrid)
                self.debug('x %s', x)
                Qext = 5.0 / (self.mieQ[idx] * x**(-4.0) + x**(0.2))

                # This must transform um to the xsec format in TauREx (m2)
                am = a * 1e-6

                sigma_mie = Qext * np.pi * (am**2.0)

                sigma_xsec[idx, ...] = sigma_mie * self.mieMixing[idx]
            else:
                sigma_xsec[idx, ...] = 0.0

        if np.any(sigma_xsec < 0):
            print(self.name + " has negative sigmas!")

        # For some particular combination of wl, Q, and particle radius,
        # sigma can end up negative. No bueno.
        sigma_xsec[sigma_xsec < 0] = 0.0    
        self.sigma_xsec = sigma_xsec
            
        self.debug('final xsec %s', self.sigma_xsec)

        yield 'Lee', sigma_xsec

@njit
def calctau(Q0, am, wlm, kwlm, tau):
    x = 2 * np.pi * am / wlm
    xkwl = 2 * np.pi * am / kwlm
    denom = Q0 * x**-4 + x**0.2
    numer = Q0 * xkwl**-4 + xkwl**0.2
    f = numer / denom
    return f * tau

@njit(nopython=True, nogil=True)
def contribute_tau(startK, endK, tau_wn, nlayers, ngrid, layer, tau):
    for k in range(startK, endK):
        for wn in range(ngrid):
            tau[layer, wn] += tau_wn[k+layer,wn]
        
class CloudTauContribution(taurex.contributions.Contribution):
    '''
    A special Contribution class for forward models where the optical
    depth of the clouds are known at a single wavelength (e.g.,
    a double-gray GCM)
    '''
    def __init__(self, tau=None, Q0=10, particle_size=0.1,
                 knowntauwl=5.0, name=''):
        super().__init__(name)
        self.tau        = tau
        self.knowntauwl = knowntauwl
        self.Q0         = Q0
        self.a          = particle_size

    def prepare_each(self, model, wngrid):
        '''
        Function needed by TauREx to do any preparatory calculation
        for the contribution. Generally, should yield the contribution
        name and the 'sigma' cross section of the contribution. 
        In this case, we have tau already, so we add the
        wavelength dependence, yield that, and modify .contribute() 
        accordingly.
        '''
        tau_wn = np.zeros(shape=(self._nlayers, wngrid.shape[0]))

        # Compute tau's wavelength dependence using ratio of
        # tau at the known wavelength to generic tau
        wlm = 10000 / wngrid * 1e-6
        kwlm = self.knowntauwl * 1e-6
        am  = self.a * 1e-6
        
        for k in range(self._nlayers):
            tau_wn[k] = calctau(self.Q0, am[k], wlm, kwlm, self.tau[k])

        self.tau_wn = tau_wn
        
        yield 'Cloud Tau', tau_wn

    def contribute(self, model, start_layer, end_layer,
                   density_offset, layer, density, tau, path_length=None):
        contribute_tau(start_layer, end_layer, self.tau_wn, self._nlayers,
                       self._ngrid, layer, tau)
