import numpy as np
import time
import theano

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

def specgrid(params, fit):
    """
    Calculate emission from each cell of a planetary grid, as a 
    fraction of stellar flux, NOT
    accounting for visibility. Observer is assumed to be looking
    directly at each grid cell. For efficiency, never-visible cells
    are not calculated. Function returns a spectrum of zeros for those
    grid cells.
    """
    cfg = fit.cfg

    # Initialize to a list because we don't know the native wavenumber
    # resolution a priori of creating the model
    nlat, nlon = fit.lat.shape
    fluxgrid = np.empty((nlat, nlon), dtype=list)
    
    if cfg.mapfunc == 'constant':
        tgrid, p = atm.tgrid(cfg.nlayers, cfg.res, fit.tmaps,
                             10.**params, cfg.pbot, cfg.ptop,
                             kind='linear', oob=cfg.oob)

        #tgrid[:,:,:] = 1000.

        r, p, abn, spec = atm.atminit(cfg.atmtype, cfg.atmfile,
                                      p, tgrid,
                                      cfg.planet.m, cfg.planet.r,
                                      cfg.planet.p0, cfg.elemfile,
                                      cfg.outdir)
    else:
        print("ERROR: Unrecognized/unimplemented map function.")

    # Determine which grid cells to use
    # Only considers longitudes currently
    nlat, nlon = fit.lat.shape
    ilat, ilon = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                          (fit.lon - fit.dlon / 2. < fit.maxvislon))
    
    if cfg.rtfunc == 'taurex':
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
            nlayers=cfg.nlayers,
            atm_min_pressure=cfg.ptop * 1e5,
            atm_max_pressure=cfg.pbot * 1e5)
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
            rtchem = taurex.chemistry.TaurexChemistry()
            for k in range(len(spec)):
                if (spec[k] not in ['H2', 'He']) and \
                   (spec[k]     in fit.cfg.cfg['taurex']['mols'].split()):
                    gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                    rtchem.addGas(gas)
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

            wn, flux, tau, ex = rt.model(wngrid=fit.wngrid)

            fluxgrid[i,j] = flux

        # Fill in non-visible cells with zeros
        # (np.where doesn't work because of broadcasting issues)
        for i in range(nlat):
            for j in range(nlon):
                if type(fluxgrid[i,j]) == type(None):
                    fluxgrid[i,j] = np.zeros(len(wn))

        # Convert to 3d array (rather than 2d array of arrays)
        fluxgrid = np.array(fluxgrid)

    else:
        print("ERROR: Unrecognized RT function.")

    return fluxgrid, wn
                                        
def specvtime(params, fit, system):
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
    nfilt = len(fit.cfg.filtfiles)

    # Integrate to filters
    intfluxgrid = np.zeros((nlat, nlon, nfilt))

    for i in range(nlat):
        for j in range(nlon):
            intfluxgrid[i,j] = utils.specint(wn, fluxgrid[i,j],
                                             fit.filtwn, fit.filttrans)
    print("Spectrum integration: {} seconds".format(time.time() - tic))
    tic = time.time()

    fluxvtime = np.zeros((nfilt, nt))

    # Account for vis and sum over grid cells
    for it in range(nt):
        for ifilt in range(nfilt):
            fluxvtime[ifilt,it] += np.sum(intfluxgrid[:,:,ifilt] * fit.vis[it])

    print("Visibility calculation: {} seconds".format(time.time() - tic))
    return fluxvtime

def sysflux(params, fit, system):
    # Calculate Fp/Fs
    fpfs    = specvtime(params, fit, system)
    nfilt, nt = fpfs.shape
    systemflux = np.zeros((nfilt, nt))
    # Account for stellar correction
    # Transform fp/fs -> fp/(fs + corr) -> (fp + fs + corr)/(fs + corr)
    for i in range(nfilt):
        fpfscorr = fpfs[i] * fit.sflux / (fit.sflux + fit.scorr[i])
        systemflux[i] = fpfscorr + 1
    return systemflux.flatten()

    
