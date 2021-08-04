import numpy as np
import time
import theano
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import mc3
import gc
from numba import jit

# Lib imports
import cf
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
            f = np.ones(len(t)) * np.min(totint)
            return f

    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * params[i]
   
    f += params[i+1] * y00

    f += params[i+2]

    f += sflux

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

    # Determine which grid cells to use
    # Only considers longitudes currently
    nlat, nlon = fit.lat.shape
    ilat, ilon = fit.ivislat, fit.ivislon

    # Initialize to a list because we don't know the native wavenumber
    # resolution a priori of creating the model
    nlat, nlon = fit.lat.shape
    fluxgrid = np.empty((nlat, nlon), dtype=list)
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
                nlayers=cfg.threed.nlayers)
            rt.add_contribution(taurex.contributions.AbsorptionContribution())
            rt.add_contribution(taurex.contributions.CIAContribution())
            if 'H-' in fit.cfg.threed.mols:
                rt.add_contribution(trc.HMinusContribution())

            rt.build()

            # If we have negative temperatures, don't run the model
            # (it will fail). Return a bad fit instead. 
            if negativeT:
                fluxgrid = -1 * np.ones((nlat, nlon,
                                         len(rt.nativeWavenumberGrid)))
                return fluxgrid, rt.nativeWavenumberGrid
            
            wn, flux, tau, ex = rt.model(wngrid=fit.wngrid)

            fluxgrid[i,j] = flux
            taugrid[i,j] = tau

        # Fill in non-visible cells with zeros
        # (np.where doesn't work because of broadcasting issues)
        nwn = len(wn)
        for i in range(nlat):
            for j in range(nlon):
                if type(fluxgrid[i,j]) == type(None):
                    fluxgrid[i,j] = np.zeros(nwn)
                if type(taugrid[i,j]) == type(None):
                    taugrid[i,j] = np.zeros((cfg.threed.nlayers, nwn))

    else:
        print("ERROR: Unrecognized RT function.")       

    return fluxgrid, tgrid, taugrid, p, wn, pmaps
                                        
def specvtime(params, fit):
    """
    Calculate spectra emitted by each grid cell, integrate over filters,
    account for line-of-sight and stellar visibility (as functions of time),
    and sum over the grid cells. Returns an array of (nfilt, nt). Units
    are fraction of stellar flux, Fp/Fs.
    """
    tic = time.time()
    # Calculate grid of spectra without visibility correction
    fluxgrid, tgrid, taugrid, p, wn, pmaps = specgrid(params, fit)
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

    fluxvtime = np.zeros((nfilt, nt))

    # Account for vis and sum over grid cells
    for it in range(nt):
        for ifilt in range(nfilt):
            fluxvtime[ifilt,it] = np.sum(intfluxgrid[:,:,ifilt] * fit.vis[it])

    # There is a very small memory leak somewhere, but this seems to
    # fix it. Not an elegant solution, but ¯\_(ツ)_/¯
    gc.collect()

    return fluxvtime, tgrid, taugrid, p, wn, pmaps

def sysflux(params, fit):
    # Calculate Fp/Fs
    fpfs, tgrid, taugrid, p, wn, pmaps = specvtime(params, fit)
    nfilt, nt = fpfs.shape
    systemflux = np.zeros((nfilt, nt))
    # Account for stellar correction
    # Transform fp/fs -> fp/(fs + corr) -> (fp + fs + corr)/(fs + corr)
    for i in range(nfilt):
        fpfscorr = fpfs[i] * fit.sflux / (fit.sflux + fit.scorr[i])
        systemflux[i] = fpfscorr + 1
    return systemflux.flatten(), tgrid, taugrid, p, wn, pmaps

def mcmc_wrapper(params, fit):
    systemflux, tgrid, taugrid, p, wn, pmaps = sysflux(params, fit)

    # Integrate cf if asked for
    if fit.cfg.threed.fitcf:
        cfsd = cfsigdiff(fit, tgrid, wn, taugrid, p, pmaps)
        return np.concatenate((systemflux, cfsd))
    
    else:
        return systemflux

def cfsigdiff(fit, tgrid, wn, taugrid, p, pmaps):
    '''
    Computes the distance between a 2D pressure/temperature map
    and the corresponding contribution function, in units of 
    "sigma". Sigma is estimated by finding the 68.3% credible region of
    the contribution function and calculating the +/- distances from
    the edges of this region to the pressure of maximum contribution.
    The sigma distance is computed for every visible grid cell
    and returned in a flattened array.
    '''
    cfs = cf.contribution_filters(tgrid, wn, taugrid, p, fit.filtwn,
                                  fit.filttrans)
    tic = time.time()
    # Where the maps "should" be
    # Find the roots of the derivative of a spline fit to
    # the contribution functions, then calculate some sort
    # of goodness of fit
    nlev, nlat, nlon = tgrid.shape
    nfilt = len(fit.cfg.twod.filtfiles)
    cfsigdiff = np.zeros(nfilt * fit.ivislat.size)
    logp = np.log10(p)
    order = np.argsort(logp)

    # Where to interpolate later
    xpdf = np.linspace(np.amin(logp),
                       np.amax(logp),
                       10*len(logp))
    count = 0
    for i, j in zip(fit.ivislat, fit.ivislon):
        for k in range(nfilt):
            # Where the map is placed
            xval = np.log10(pmaps[k,i,j])

            # Interpolate CF to 10x finer atmospheric layers
            pdf = np.interp(xpdf, logp[order], cfs[i,j,order,k])

            # Compute minimum density of 68.3% region
            pdf, xpdf, HPDmin = mc3.stats.cred_region(pdf=pdf, xpdf=xpdf)

            # Calculate 68.3% boundaries
            siglo = np.amin(xpdf[pdf>HPDmin])
            sighi = np.amax(xpdf[pdf>HPDmin])

            # Assume CF is approx. Gaussian
            xpeak = (sighi + siglo) / 2
            sig   = (sighi - siglo) / 2

            cfsigdiff[count] = (xval - xpeak) / sig
            count += 1

    return cfsigdiff

def get_par(fit):
    '''
    Returns sensible parameter settings for each model
    '''
    nmaps = len(fit.maps)
    if fit.cfg.threed.mapfunc == 'isobaric':
        npar  = nmaps
        # Guess that higher temps are deeper
        ipar  = np.argsort(np.max(fit.tmaps, axis=(1,2)))
        par   = np.linspace(-2, 0, npar)[ipar]
        pstep = np.ones(npar) * 1e-3
        pmin  = np.ones(npar) * np.log10(fit.cfg.threed.ptop)
        pmax  = np.ones(npar) * np.log10(fit.cfg.threed.pbot)
        pnames = ['log(p{})'.format(a) for a in np.arange(1,nmaps+1)]
    elif fit.cfg.threed.mapfunc == 'sinusoidal':
        # For a single wavelength
        npar = 4
        par   = np.zeros(npar)
        pstep = np.ones(npar) * 1e-3
        pmin  = np.array([np.log10(fit.cfg.threed.ptop),
                          -np.inf, -np.inf, -180.0])
        pmax  = np.array([np.log10(fit.cfg.threed.pbot),
                          np.inf,  np.inf,  180.0])
        pnames = ['log(p{})',
                  'Lat. Amp. {}',
                  'Lon. Amp. {}',
                  'Lon. Phase {}']
        # Repeat for each wavelength
        nwl = len(fit.maps)
        par   = np.tile(par,   nwl)
        pstep = np.tile(pstep, nwl)
        pmin  = np.tile(pmin,  nwl)
        pmax  = np.tile(pmax,  nwl)
        pnames = np.concatenate([[pname.format(a) for pname in pnames] \
                                 for a in np.arange(1, nmaps+1)]) # Trust me
        # Guess that longitudinal sinusoid follows the hotpost
        for i in range(nwl):
            par[3+i*npar] = fit.maps[i].hslocbest[1]
        # Guess that higher temps are deeper
        ipar = np.argsort(np.max(fit.tmaps, axis=(1,2)))
        for i in range(nwl):
            par[i*npar]   = np.linspace(-2, 0, nwl)[ipar][i]
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
        pnames = ['log(p{},{},{})'.format(i,j,k) \
                  for i in np.arange(1, nmaps+1) \
                  for j in ilat \
                  for k in ilon]
    elif fit.cfg.threed.mapfunc == 'quadratic':
        # For a single wavelength
        npar  = 6
        par   = np.zeros(npar)
        pstep = np.ones(npar) * 1e-3
        pmin  = np.array([np.log10(fit.cfg.threed.ptop),
                          -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        pmax  = np.array([np.log10(fit.cfg.threed.pbot),
                          np.inf, np.inf, np.inf, np.inf, np.inf])
        pnames = ['log(p{})',
                  'LatLat {}',
                  'LonLon {}',
                  'Lat {}',
                  'Lon {}',
                  'LatLon {}']
        # Repeat for each wavelength
        nwl = len(fit.maps)
        par   = np.tile(par,   nwl)
        pstep = np.tile(pstep, nwl)
        pmin  = np.tile(pmin,  nwl)
        pmax  = np.tile(pmax,  nwl)
        pnames = np.concatenate([[pname.format(a) for pname in pnames] \
                                 for a in np.arange(1, nmaps+1)]) # Trust me
    elif fit.cfg.threed.mapfunc == 'cubic':
        # For a single wavelength
        npar  = 10
        par   = np.zeros(npar)
        pstep = np.ones(npar) * 1e-3
        pmin  = np.array([np.log10(fit.cfg.threed.ptop),
                          -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                          -np.inf, -np.inf, -np.inf, -np.inf])
        pmax  = np.array([np.log10(fit.cfg.threed.pbot),
                          np.inf, np.inf, np.inf, np.inf, np.inf,
                          np.inf, np.inf, np.inf, np.inf])
        pnames = ['log(p{})',
                  'LatLatLat {}',
                  'LonLonLon {}',
                  'LatLat {}',
                  'LonLon {}',
                  'Lat {}',
                  'Lon {}',
                  'LatLatLon {}',
                  'LatLonLon {}',
                  'LatLon {}']
        # Repeat for each wavelength
        nwl = len(fit.maps)
        par   = np.tile(par,   nwl)
        pstep = np.tile(pstep, nwl)
        pmin  = np.tile(pmin,  nwl)
        pmax  = np.tile(pmax,  nwl)
        pnames = np.concatenate([[pname.format(a) for pname in pnames] \
                                 for a in np.arange(1, nmaps+1)]) # Trust me
    else:
        print("Warning: Unrecognized mapping function.")

    if fit.cfg.threed.oob == 'both':
        par    = np.concatenate((par,   (1000., 2000.)))
        pstep  = np.concatenate((pstep, (   1.,    1.)))
        pmin   = np.concatenate((pmin,  (   0.,    0.)))
        pmax   = np.concatenate((pmax,  (4000., 4000.)))
        pnames = np.concatenate((pnames, ('Ttop', 'Tbot')))
    elif fit.cfg.threed.oob == 'top':
        par    = np.concatenate((par,   (1000.,)))
        pstep  = np.concatenate((pstep, (   1.,)))
        pmin   = np.concatenate((pmin,  (   0.,)))
        pmax   = np.concatenate((pmax,  (4000.,)))
        pnames = np.concatenate((pnames, ('Ttop',)))
    elif fit.cfg.threed.oob == 'bot':
        par    = np.concatenate((par,   (2000.,)))
        pstep  = np.concatenate((pstep, (   1.,)))
        pmin   = np.concatenate((pmin,  (   0.,)))
        pmax   = np.concatenate((pmax,  (4000.,)))
        pnames = np.concatenate((pnames, ('Tbot',)))
    else:
        print("Unrecognized out-of-bounds rule.")

    return par, pstep, pmin, pmax, pnames
        
    
