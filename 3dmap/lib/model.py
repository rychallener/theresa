import numpy as np
import time

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

    # Initialize to a list because we don't know the native wavenumber
    # resolution a priori of creating the model
    fluxgrid = []
    latgrid  = []
    longrid  = []
    
    if cfg.mapfunc == 'constant':
        tgrid, p = atm.tgrid(cfg.nlayers, cfg.res, fit.tmaps,
                             10.**params, cfg.pbot, cfg.ptop,
                             kind='linear', oob=cfg.oob)

        r, p, abn, spec = atm.atminit(cfg.atmtype, cfg.atmfile,
                                      p, tgrid,
                                      cfg.planet.m, cfg.planet.r,
                                      cfg.planet.p0, cfg.elemfile,
                                      cfg.outdir)
    else:
        print("ERROR: Unrecognized/unimplemented map function.")

    # Determine which grid cells to use
    # Only considers longitudes currently
    ilat, ilon = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                          (fit.lon - fit.dlon / 2. < fit.maxvislon))
    
    if cfg.rtfunc == 'taurex':
        # Latitudes (all visible) and Longitudes
        for i, j in zip(ilat, ilon):
            # Check for nonphysical atmosphere and return a bad fit
            # if so
            if not np.all(tgrid[:,i,j] >= 0):
                msg = "WARNING: Nonphysical TP profile at Lat: {}, Lon: {}"
                print(msg.format(fit.lat[i,j], fit.lon[i,j]))
                return np.ones(len(cfg.filtfiles)) * -1
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
                latmin=fit.lat[i,j] - fit.dlat / 2.,
                latmax=fit.lat[i,j] + fit.dlat / 2.,
                lonmin=fit.lon[i,j] - fit.dlon / 2.,
                lonmax=fit.lon[i,j] + fit.dlon / 2.)
            rt.add_contribution(taurex.contributions.AbsorptionContribution())
            rt.add_contribution(taurex.contributions.CIAContribution())
            rt.build()

            wn, flux, tau, ex = rt.model()
            fluxgrid.append(flux)
            latgrid.append(fit.lat[i,j])
            longrid.append(fit.lon[i,j])
            
        fluxgrid = np.array(fluxgrid)
        latgrid  = np.array(latgrid)
        longrid  = np.array(longrid)

    else:
        print("ERROR: Unrecognized RT function.")

    return fluxgrid, wn, latgrid, longrid
                                        
def fit_spec_all(params, fit, planet, system):
    tic = time.time()
    # Calculate grid of spectra without visibility correction
    fluxgrid, wn, latgrid, longrid = fit_spec(params, fit)
    print("Spectrum generation: {} seconds".format(time.time() - tic))
    tic = time.time()

    latgrid *= np.pi / 180.
    longrid *= np.pi / 180.

    nt    = len(fit.t)
    ngrid = len(latgrid)
    nfilt = len(fit.cfg.filtfiles)

    # Integrate to filters
    intfluxgrid = np.zeros((ngrid, nfilt))

    for i in range(ngrid):
        intfluxgrid[i] = utils.specint(wn, fluxgrid[i], fit.cfg.filtfiles)

    fluxvtime = np.zeros((nfilt, nt))

    theta0 = system.secondaries[0].theta0.eval()
    prot   = system.secondaries[0].prot.eval()
    t0     = system.secondaries[0].t0.eval()
    rp     = system.secondaries[0].r.eval()
    rs     = system.primary.r.eval()
    
    x, y, z = system.position(fit.t)
    x, y = x.eval(), y.eval()

    # Calculate visibility of each grid cell based on observer LoS
    # (i.e., from observer PoV, where centlon is 100% visible)
    for it in range(nt):
        print("Time index: {}".format(it))
        tic2 = time.time()
        vis = utils.visibility(fit.t[it], latgrid, longrid, fit.dlat,
                               fit.dlon, theta0, prot, t0, rp, rs,
                               x[:,it], y[:,it])
        print("For this index: {} seconds".format(time.time() - tic2))

        # Account for vis and sum over grid cells
        for igrid in range(ngrid):
            fluxvtime[:,it] += intfluxgrid[igrid] * vis[igrid]

    print("Visibility calculation: {} seconds".format(time.time() - tic))
    return fluxvtime
                        
                    
                
        

    
