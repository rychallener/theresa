import numpy as np

# Lib imports
import atm
import utils
import constants as c
import taurexclass as trc

# Taurex imports
import taurex
from taurex import chemistry
from taurex import planet
from taurex import stellar
from taurex import model
from taurex import pressure
from taurex import temperature
from taurex import cache
from taurex import contributions
from taurex import optimizer
# This import is explicit because it's not included in taurex.temperature. Bug?
from taurex.data.profiles.temperature.temparray import TemperatureArray

def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens):
    """
    Basic 2D fitting routine for a single wavelength.
    """
    # Check for negative intensities
    if type(intens) != type(None):
        nloc = intens.shape[1]
        for j in range(nloc):
            # Weighted eigenmap intensity
            totint = np.sum(intens[:,j] * params[:ncurves])
            # Contribution from uniform map
            totint += params[ncurves] / np.pi
            if totint <= 0:
                f = np.ones(len(t)) * -1
                return f

    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * params[i]
   
    f += params[i+1] * y00

    f += params[i+2]

    f += sflux

    return f

def fit_2d_wl(params, ecurves, t, wl, y00, sflux, ncurves, intens):
    """
    2D fitting driver that calls the 2D fitting routine for each
    wavelength.
    """
    f = np.zeros(len(t) * len(wl))

    nt   = len(t)
    nw   = len(wl)
    npar = int(len(params) / nw) # params per wavelength
    for i in range(nw):            
        f[i*nt:(i+1)*nt] = fit_2d(params[i*npar:(i+1)*npar], ecurves,
                                  t, y00, sflux, ncurves, intens)

    return f

def fit_spec(params, fit):
    """
    Fit a single spectrum.
    """
    cfg = fit.cfg
    
    if cfg.mapfunc == 'constant':
        tgrid, p = atm.tgrid(cfg.nlayers, cfg.res, fit.tmaps,
                             params, cfg.pbot, cfg.ptop,
                             kind='linear', bounds_error=False,
                             fill_value='extrapolate')

        r, p, abn, spec = atm.atminit(cfg.atmtype, cfg.atmfile,
                                      p, tgrid,
                                      cfg.planet.m, cfg.planet.r,
                                      cfg.planet.p0, cfg.elemfile,
                                      cfg.outdir)
    else:
        print("ERROR: Unrecognized/unimplemented map function.")
    
    if cfg.rtfunc == 'taurex':
        for i in range(cfg.res):
            for j in range(cfg.res):
                rtt = TemperatureArray(
                    tp_array=tgrid[:,i,j])
                rtplan = taurex.planet.Planet(
                    planet_mass=cfg.planet.m*c.Msun/c.Mjup,
                    planet_radius=cfg.planet.r*c.Rsun/c.Rjup,
                    planet_distance=cfg.planet.a,
                    impact_param=cfg.planet.b,
                    orbital_period=cfg.planet.porb,
                    transit_time=cfg.planet.t0)
                rtstar = taurex.stellar.Star(
                    temperature=cfg.star.t,
                    radius=cfg.star.r,
                    distance=cfg.star.d,
                    metallicity=cfg.star.z)
                rtchem = taurex.chemistry.TaurexChemistry()
                for k in range(len(spec)):
                    if spec[k] not in ['H2', 'He']:
                        gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                        rtchem.addGas(gas)
                rtp = taurex.pressure.SimplePressureProfile(
                    nlayers=cfg.nlayers,
                    atm_min_pressure=cfg.ptop * 1e5,
                    atm_max_pressure=cfg.pbot * 1e5)
                rt = trc.EmissionModel3D(
                    planet=rtplan,
                    star=rtstar,
                    pressure_profile=rtp,
                    temperature_profile=rtt,
                    chemistry=rtchem,
                    nlayers=cfg.nlayers,
                    latmin=fit.lat[i,j],
                    latmax=fit.lat[i,j] + fit.dlat,
                    lonmin=fit.lon[i,j],
                    lonmax=fit.lon[i,j] + fit.dlon)
                rt.add_contribution(taurex.contributions.AbsorptionContribution())
                rt.add_contribution(taurex.contributions.CIAContribution())
                rt.build()
                if i == 0 and j == 0:
                    flux = np.zeros((len(rt.nativeWavenumberGrid),
                                     cfg.res, cfg.res))
                wn, flux[:,i,j], tau, ex = rt.model()

        totflux = np.sum(flux, axis=(1,2))

        inttotflux = utils.specint(wn, totflux, cfg.filtfiles)

    else:
        print("ERROR: Unrecognized RT function.")

    return inttotflux
                                        
    
    
