import numpy as np
import time
import theano
from numba import jit

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

@jit(nopython=True)
def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens):
    """
    Basic 2D fitting routine for a single wavelength.
    """
    # Check for negative intensities
    if intens is not None:
        nloc = intens.shape[1]
        totint = np.zeros(nloc)
        for j in range(nloc):
            # Weighted eigenmap intensity
            totint[j] = np.sum(intens[:,j] * params[:ncurves])
            # Contribution from uniform map
            totint[j] += params[ncurves] / np.pi
        if np.any(totint <= 0):
            #print("Negative at loc {}.".format(j))
            f = np.ones(len(t)) * np.min(totint)
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

def specgrid(params, fit, return_tau=False):
    """
    Calculate emission from each cell of a planetary grid, as a 
    fraction of stellar flux, NOT
    accounting for visibility. Observer is assumed to be looking
    directly at each grid cell. For efficiency, never-visible cells
    are not calculated. Function returns a spectrum of zeros for those
    grid cells.
    """
    cfg = fit.cfg

    # Determine which grid cells to use
    # Only considers longitudes currently
    nlat, nlon = fit.lat.shape
    ilat, ilon = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                          (fit.lon - fit.dlon / 2. < fit.maxvislon))

    # Initialize to a list because we don't know the native wavenumber
    # resolution a priori of creating the model
    nlat, nlon = fit.lat.shape
    fluxgrid = np.empty((nlat, nlon), dtype=list)
    if return_tau:
        taugrid = np.empty((nlat, nlon), dtype=list)

    pmaps = atm.pmaps(params, fit)
    tgrid, p = atm.tgrid(cfg.threed.nlayers, cfg.twod.nlat,
                         cfg.twod.nlon, fit.tmaps, pmaps,
                         cfg.threed.pbot, cfg.threed.ptop, params,
                         interptype=cfg.threed.interp,
                         oob=cfg.threed.oob, smooth=cfg.threed.smooth)

    abn, spec = atm.atminit(cfg.threed.atmtype, cfg.threed.mols, p,
                            tgrid, cfg.planet.m, cfg.planet.r,
                            cfg.planet.p0, cfg.threed.elemfile,
                            cfg.outdir, ilat=ilat, ilon=ilon,
                            cheminfo=fit.cheminfo)

    negativeT = False
    
    if cfg.threed.rtfunc == 'taurex':
        # Cell-independent Tau-REx objects
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
        rtp = taurex.pressure.SimplePressureProfile(
            nlayers=cfg.threed.nlayers,
            atm_min_pressure=cfg.threed.ptop * 1e5,
            atm_max_pressure=cfg.threed.pbot * 1e5)
        # Latitudes (all visible) and Longitudes
        for i, j in zip(ilat, ilon):
            # Check for nonphysical atmosphere and return a bad fit
            # if so
            if not np.all(tgrid[:,i,j] >= 0):
                msg = "WARNING: Nonphysical TP profile at Lat: {}, Lon: {}"
                print(msg.format(fit.lat[i,j], fit.lon[i,j]))
                negativeT = True
            rtt = TemperatureArray(
                tp_array=tgrid[:,i,j])
            rtchem = taurex.chemistry.TaurexChemistry()
            for k in range(len(spec)):
                if (spec[k] not in ['H2', 'He']) and \
                   (spec[k]     in fit.cfg.threed.mols):
                    gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                    rtchem.addGas(gas)
            rt = trc.EmissionModel3D(
                planet=rtplan,
                star=rtstar,
                pressure_profile=rtp,
                temperature_profile=rtt,
                chemistry=rtchem,
                nlayers=cfg.threed.nlayers,
                latmin=fit.lat[i,j] - fit.dlat / 2.,
                latmax=fit.lat[i,j] + fit.dlat / 2.,
                lonmin=fit.lon[i,j] - fit.dlon / 2.,
                lonmax=fit.lon[i,j] + fit.dlon / 2.)
            rt.add_contribution(taurex.contributions.AbsorptionContribution())
            rt.add_contribution(taurex.contributions.CIAContribution())

            rt.build()

            # If we have negative temperatures, don't run the model
            # (it will fail). Return a bad fit instead. 
            if negativeT:
                fluxgrid = -1 * np.ones((nlat, nlon,
                                         len(rt.nativeWavenumberGrid)))
                return fluxgrid, rt.nativeWavenumberGrid
            
            wn, flux, tau, ex = rt.model(wngrid=fit.wngrid)

            fluxgrid[i,j] = flux
            if return_tau:
                taugrid[i,j] = tau

        # Fill in non-visible cells with zeros
        # (np.where doesn't work because of broadcasting issues)
        nwn = len(wn)
        for i in range(nlat):
            for j in range(nlon):
                if type(fluxgrid[i,j]) == type(None):
                    fluxgrid[i,j] = np.zeros(nwn)
                if return_tau:
                    if type(taugrid[i,j]) == type(None):
                        taugrid[i,j] = np.zeros((cfg.threed.nlayers, nwn))

        # Convert to 3d array (rather than 2d array of arrays)
        fluxgrid = np.concatenate(np.concatenate(fluxgrid)).reshape(nlat,
                                                                    nlon,
                                                                    nwn)

    else:
        print("ERROR: Unrecognized RT function.")

    if return_tau:
        return fluxgrid, wn, taugrid
    
    return fluxgrid, wn
                                        
def specvtime(params, fit):
    """
    Calculate spectra emitted by each grid cell, integrate over filters,
    account for line-of-sight and stellar visibility (as functiosn of time),
    and sum over the grid cells. Returns an array of (nfilt, nt). Units
    are fraction of stellar flux, Fp/Fs.
    """
    tic = time.time()
    # Calculate grid of spectra without visibility correction
    fluxgrid, wn = specgrid(params, fit)
    print("Spectrum generation: {} seconds".format(time.time() - tic))
    tic = time.time()

    nt         = len(fit.t)
    nlat, nlon = fit.lat.shape
    nfilt = len(fit.cfg.twod.filtfiles)

    # Integrate to filters
    intfluxgrid = np.zeros((nlat, nlon, nfilt))

    for i in range(nlat):
        for j in range(nlon):
            intfluxgrid[i,j] = utils.specint(wn, fluxgrid[i,j],
                                             fit.filtwn, fit.filttrans)
    #print("Spectrum integration: {} seconds".format(time.time() - tic))
    tic = time.time()

    fluxvtime = np.zeros((nfilt, nt))

    # Account for vis and sum over grid cells
    for it in range(nt):
        for ifilt in range(nfilt):
            fluxvtime[ifilt,it] = np.sum(intfluxgrid[:,:,ifilt] * fit.vis[it])

    #print("Visibility calculation: {} seconds".format(time.time() - tic))
    return fluxvtime

def sysflux(params, fit):
    # Calculate Fp/Fs
    fpfs    = specvtime(params, fit)
    nfilt, nt = fpfs.shape
    systemflux = np.zeros((nfilt, nt))
    # Account for stellar correction
    # Transform fp/fs -> fp/(fs + corr) -> (fp + fs + corr)/(fs + corr)
    for i in range(nfilt):
        fpfscorr = fpfs[i] * fit.sflux / (fit.sflux + fit.scorr[i])
        systemflux[i] = fpfscorr + 1
    return systemflux.flatten()

def get_par(fit):
    '''
    Returns sensible parameter settings for each model
    '''
    if fit.cfg.threed.mapfunc == 'isobaric':
        npar  = len(fit.maps)
        # Guess that higher temps are deeper
        ipar  = np.argsort(np.max(fit.tmaps, axis=(1,2)))
        par   = np.linspace(-2, 0, npar)[ipar]
        pstep = np.ones(npar) * 1e-3
        pmin  = np.ones(npar) * np.log10(fit.cfg.threed.ptop)
        pmax  = np.ones(npar) * np.log10(fit.cfg.threed.pbot)
    elif fit.cfg.threed.mapfunc == 'sinusoidal':
        # For a single wavelength
        npar = 4
        par   = np.array([0.0, -10.0, -10.0, 30.0])
        pstep = np.ones(npar) * 1e-3
        pmin  = np.array([np.log10(fit.cfg.threed.ptop),
                          -np.inf, -np.inf, -180.0])
        pmax  = np.array([np.log10(fit.cfg.threed.pbot),
                          np.inf,  np.inf,  180.0])
        # Repeat for each wavelength
        nwl = len(fit.maps)
        par   = np.tile(par,   nwl)
        pstep = np.tile(pstep, nwl)
        pmin  = np.tile(pmin,  nwl)
        pmax  = np.tile(pmax,  nwl)
    elif fit.cfg.threed.mapfunc == 'flexible':
        ilat, ilon = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                              (fit.lon - fit.dlon / 2. < fit.maxvislon))
        nvislat = len(np.unique(ilat))
        nvislon = len(np.unique(ilon))
        npar = nvislat * nvislon * len(fit.maps)
        par   = np.zeros(npar)
        pstep = np.ones(npar) * 1e-3
        pmin  = np.ones(npar) * np.log10(fit.cfg.threed.ptop)
        pmax  = np.ones(npar) * np.log10(fit.cfg.threed.pbot)
    else:
        print("Warning: Unrecognized mapping function.")

    if fit.cfg.threed.oob == 'both':
        par   = np.concatenate((par,   (1000., 2000.)))
        pstep = np.concatenate((pstep, (   1.,    1.)))
        pmin  = np.concatenate((pmin,  (   0.,    0.)))
        pmax  = np.concatenate((pmax,  (4000., 4000.)))
    elif fit.cfg.threed.oob == 'top':
        par   = np.concatenate((par,   (1000.,)))
        pstep = np.concatenate((pstep, (   1.,)))
        pmin  = np.concatenate((pmin,  (   0.,)))
        pmax  = np.concatenate((pmax,  (4000.,)))
    elif fit.cfg.threed.oob == 'bot':
        par   = np.concatenate((par,   (2000.,)))
        pstep = np.concatenate((pstep, (   1.,)))
        pmin  = np.concatenate((pmin,  (   0.,)))
        pmax  = np.concatenate((pmax,  (4000.,)))
    else:
        print("Unrecognized out-of-bounds rule.")

    return par, pstep, pmin, pmax
        
    