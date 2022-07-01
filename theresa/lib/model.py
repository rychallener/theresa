import numpy as np
import time
import theano
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import mc3
import gc
import sys
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
def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens, baseline):
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

    if baseline == 'linear':
        f += params[i+3] * (t - params[i+4])
    elif baseline == 'quadratic':
        f += params[i+3] * (t - params[i+5])**2 + \
             params[i+4] * (t - params[i+5])

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
                         fit.nparams3d, fit.modeltype3d, fit.imodel3d,
                         interptype=cfg.threed.interp,
                         smooth=cfg.threed.smooth)
    
    if cfg.threed.z == 'fit':
        izmodel = np.where(fit.modeltype3d == 'z')[0][0]
        istart = np.sum(fit.nparams3d[:izmodel])
        z = params[istart]
    else:
        z = cfg.threed.z

    mols = np.concatenate((cfg.threed.mols, cfg.threed.cmols))
    abn, spec = atm.atminit(cfg.threed.atmtype, mols, p, tgrid,
                            cfg.planet.m, cfg.planet.r, cfg.planet.p0,
                            z, ilat=ilat, ilon=ilon,
                            cheminfo=fit.cheminfo)
    
    negativeT = False

    # Set up cloud grid(s)
    if 'clouds' in fit.modeltype3d:
        # Cloud radius list, cloud mix ratio list, Q0 list
        crl, cml, ql = atm.cloudmodel_to_grid(fit, p, params, abn, spec)
    
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
            if 'clouds' in fit.modeltype3d:
                for icloud in range(len(crl)):
                    rt.add_contribution(
                        trc.LeeMieVaryMixContribution(
                            lee_mie_radius=crl[icloud][:,i,j],
                            lee_mie_q=ql[icloud][:,i,j],
                            lee_mie_mix_ratio=10.**cml[icloud][:,i,j],
                            lee_mie_bottomP=-1,
                            lee_mie_topP=-1))
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
    # Calculate grid of spectra without visibility correction
    fluxgrid, tgrid, taugrid, p, wn, pmaps = specgrid(params, fit)

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
    tic = time.time()
    systemflux, tgrid, taugrid, p, wn, pmaps = sysflux(params, fit)

    # Integrate cf if asked for
    if fit.cfg.threed.fitcf:
        cfsd = cfsigdiff(fit, tgrid, wn, taugrid, p, pmaps)
        print("Model Evaluation: {} s".format(time.time() - tic))
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

def get_par_2d(fit, ln):
    '''
    Returns sensible parameter settings for each 2D model
    '''
    cfg = fit.cfg
    
    # Necessary parameters
    npar = ln.ncurves + 2

    params = np.zeros(npar)
    params[ln.ncurves] = 0.001
    
    pstep = np.ones(npar) *  0.01
    pmin  = np.ones(npar) * -1.0
    pmax  = np.ones(npar) *  1.0

    pnames   = []
    texnames = []
    for j in range(ln.ncurves):
        pnames.append("C{}".format(j+1))
        texnames.append("$C_{{{}}}$".format(j+1))

    pnames.append("C0")
    texnames.append("$C_0$")

    pnames.append("scorr")
    texnames.append("$s_{corr}$")

    # Parse baseline models
    if cfg.twod.baseline is None:
        pass
    elif cfg.twod.baseline == 'linear':
        params   = np.concatenate((params,   ( 0.0,  0.0)))
        pstep    = np.concatenate((pstep,    ( 0.01, 0.0)))
        pmin     = np.concatenate((pmin,     (-1.0,  -10.0)))
        pmax     = np.concatenate((pmax,     ( 1.0,   10.0)))
        pnames   = np.concatenate((pnames,   ('b1', 't0')))
        texnames = np.concatenate((texnames, ('$b_1$', '$t_0$')))
    elif cfg.twod.baseline == 'quadratic':
        params   = np.concatenate((params,   ( 0.0,  0.0,   0.0)))
        pstep    = np.concatenate((pstep,    ( 0.01, 0.01,  0.0)))
        pmin     = np.concatenate((pmin,     (-1.0,  -1.0, -10.0)))
        pmax     = np.concatenate((pmax,     ( 1.0,   1.0,  10.0)))
        pnames   = np.concatenate((pnames,   ('b2', 'b1', 't0')))
        texnames = np.concatenate((texnames, ('$b_2$', '$b_1$', '$t_0$')))
    else:
        print("Unrecognized baseline model.")
        sys.exit()

    return params, pstep, pmin, pmax, pnames, texnames

def get_par_3d(fit):
    '''
    Returns sensible parameter settings for each 3D model.

    This function should be edited when additional models are created.
    '''
    nmaps = len(fit.maps)

    # Number of parameters for each model, in order.
    nparams   = np.zeros(len(fit.cfg.threed.modelnames), dtype=int)
    # Model type. Occasionally models are handled by their types.
    # If unsure, creating a new model type is safest.
    modeltype = []
    # List of indexing arrays which will pull out the parameters
    # of each model. E.g., params[imodel[i]] will get the parameters
    # associated with the ith model
    imodel    = []
    # Lists for parameter setting. Will be converted to 1D arrays later,
    # but we don't know their sizes a priori
    allparams = []
    allpmin   = []
    allpmax   = []
    allpstep  = []
    allpnames = []

    # Useful numbers
    logptop = np.log10(fit.cfg.threed.ptop)
    logpbot = np.log10(fit.cfg.threed.pbot)
    
    # Loops through all the given models, setting their number of
    # parameters, as well as sensible initial guesses, parameter
    # boundaries, and step sizes.
    for im, mname in enumerate(fit.cfg.threed.modelnames):   
        if mname == 'isobaric':
            npar  = nmaps
            # Guess that higher temps are deeper
            ipar  = np.argsort(np.max(fit.tmaps, axis=(1,2)))
            par   = np.linspace(-2, 0, npar)[ipar]
            pstep = np.ones(npar) * 1e-3
            pmin  = np.ones(npar) * np.log10(fit.cfg.threed.ptop)
            pmax  = np.ones(npar) * np.log10(fit.cfg.threed.pbot)
            pnames = ['log(p{})'.format(a) for a in np.arange(1,nmaps+1)]
            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'isobaric2':
            nppwl = 4
            npar  = nppwl * nmaps
            par   = np.zeros(nppwl)
            pstep = np.array([1e-3, 1e-3, 1.0, 1.0])
            pmin  = np.array([np.log10(fit.cfg.threed.ptop),
                              np.log10(fit.cfg.threed.ptop),
                              -180.0,
                              -180.0])
            pmax  = np.array([np.log10(fit.cfg.threed.pbot),
                              np.log10(fit.cfg.threed.pbot),
                              180.0,
                              180.0])
            pnames = ['log(p{})_1',
                      'log(p{})_2',
                      'W.Disc.{}',
                      'E.Disc.{}']
            # Repeat for each wavelength
            nwl = len(fit.maps)
            par   = np.tile(par,   nwl)
            pstep = np.tile(pstep, nwl)
            pmin  = np.tile(pmin,  nwl)
            pmax  = np.tile(pmax,  nwl)
            pnames = np.concatenate([[pname.format(a) for pname in pnames] \
                                     for a in np.arange(1, nmaps+1)]) # Trust me

            # Guess that higher temps are deeper
            ipar = np.argsort(np.max(fit.tmaps, axis=(1,2)))
            for i in range(nwl):
                par[i*nppwl]   = np.linspace(-2, 0, nwl)[ipar][i]
                par[i*nppwl+1] = np.linspace(-2, 0, nwl)[ipar][i]

            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'sinusoidal':
            # For a single wavelength
            nppwl = 4
            nwl   = nmaps
            npar  = nppwl * nwl
            par   = np.zeros(nppwl)
            pstep = np.ones(nppwl) * 1e-3
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
                par[3+i*nppwl] = fit.maps[i].hslocbest[1]
            # Guess that higher temps are deeper
            ipar = np.argsort(np.max(fit.tmaps, axis=(1,2)))
            for i in range(nwl):
                par[i*nppwl]   = np.linspace(-2, 0, nwl)[ipar][i]

            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'flexible':
            ilat, ilon = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                                  (fit.lon - fit.dlon / 2. < fit.maxvislon))
            nvislat = len(np.unique(ilat))
            nvislon = len(np.unique(ilon))
            nppwl = nvislat * nvislon * len(fit.maps)
            npar  = nppwl * npar
            par   = np.zeros(nppwl)
            pstep = np.ones(nppwl) * 1e-3
            pmin  = np.ones(nppwl) * np.log10(fit.cfg.threed.ptop)
            pmax  = np.ones(nppwl) * np.log10(fit.cfg.threed.pbot)
            pnames = ['log(p{},{},{})'.format(i,j,k) \
                      for i in np.arange(1, nmaps+1) \
                      for j in ilat \
                      for k in ilon]
            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'quadratic':
            # For a single wavelength
            nppwl = 6
            npar  = nppwl * nwl
            par   = np.zeros(nppwl)
            pstep = np.ones(nppwl) * 1e-3
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
            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'cubic':
            # For a single wavelength
            nppwl = 10
            npar  = nppwl * nwl
            par   = np.zeros(nppwl)
            pstep = np.ones(nppwl) * 1e-3
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
            modeltype.append('pmap')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        # Temperature profile options
        elif mname == 'ttop':
            npar   = 1
            par    = [1000.]
            pstep  = [   1.]
            pmin   = [   0.]
            pmax   = [4000.]
            pnames = ['Ttop']
            modeltype.append('ttop')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'tbot':
            npar   = 1
            par    = [2000.]
            pstep  = [   1.]
            pmin   = [ 100.]
            pmax   = [5000.]
            pnames = ['Tbot']
            modeltype.append('tbot')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        # Chemistry models
        elif mname == 'z':
            npar   = 1
            par    = [ 0.0]
            pstep  = [ 0.1]
            pmin   = [-1.0]
            pmax   = [ 1.0]
            pnames = ['z']
            modeltype.append('z')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        # Cloud models
        elif mname == 'leemie':
            npar    = 5
            # Parameters: part. size, Q0, mix ratio (log),
            #             bottom p (log), top p (log)
            logpbot = np.log10(fit.cfg.threed.pbot)
            logptop = np.log10(fit.cfg.threed.ptop)
            par     = [  0.1,   40.0, -10.0,         2.0,        -1.0]
            pstep   = [  0.1,    1.0,   1.0,         0.1,         0.1]
            pmin    = [  0.0,    0.0, -20.0, logptop - 1, logptop - 1]
            pmax    = [100.0, 1000.0,   0.0, logpbot + 1, logpbot + 1]
            pnames  = ['a', 'Q0', 'mix', 'log(cloud bottom)', 'log(cloud top)']
            modeltype.append('clouds')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'leemie2':
            npar    = 12
            # Parameters: part. size, Q0, mix ratio (log),
            #             bottom p (log), top p (log)
            logpbot = np.log10(fit.cfg.threed.pbot)
            logptop = np.log10(fit.cfg.threed.ptop)
            par     = [  0.1,   40.0, -10.0,         2.0,        -1.0,   0.1,   40.0, -10.0,         2.0,        -1.0,    0.,  180.]
            pstep   = [  0.1,    1.0,   1.0,         0.1,         0.1,   0.1,    1.0,   1.0,         0.1,         0.1,   10.,   10.]
            pmin    = [  0.0,    0.0, -20.0, logptop - 1, logptop - 1,   0.0,    0.0, -20.0, logptop - 1, logptop - 1, -180.,    0.]
            pmax    = [100.0, 1000.0,   0.0, logpbot + 1, logpbot + 1, 100.0, 1000.0,   0.0, logpbot + 1, logpbot + 1,  180.,  360.]
            pnames  = ['a1', 'Q01', 'mix1', 'log(cloud bottom)1', 'log(cloud top)1', 'a2', 'Q02', 'mix2', 'log(cloud bottom)2', 'log(cloud top)2', 'Cl.2 Center', 'Cl.2 Width']
            modeltype.append('clouds')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'leemie-clearspot':          
            npar    = 7
            # Parameters: part. size, Q0, mix ratio (log),
            #             bottom p (log), top p (log)
            logpbot = np.log10(fit.cfg.threed.pbot)
            logptop = np.log10(fit.cfg.threed.ptop)
            par     = [  0.1,   40.0, -10.0,         2.0,        -1.0,    0.0,  180.0]
            pstep   = [  0.1,    1.0,   1.0,         0.1,         0.1,   10.0,   10.0]
            pmin    = [  0.0,    0.0, -20.0, logptop - 1, logptop - 1, -180.0,    0.0]
            pmax    = [100.0, 1000.0,   0.0, logpbot + 1, logpbot + 1,  180.0,  360.0]
            pnames  = ['a', 'Q0', 'mix', 'log(cloud bottom)', 'log(cloud top)', 'Spot Center', 'Spot Width']
            modeltype.append('clouds')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        elif mname == 'eqclouds':
            npar = 2
            # Parameters: part. size, Q0
            par    = [  0.1,   40.0]
            pstep  = [  0.1,    1.0]
            pmin   = [  0.0,    0.0]
            pmax   = [100.0, 1000.0]
            pnames = ['a', 'Q0']
            modeltype.append('clouds')
            nparams[im] = npar
            allparams.append(par)
            allpmin.append(pmin)
            allpmax.append(pmax)
            allpstep.append(pstep)
            allpnames.append(pnames)
        else:
            print('WARNING: {} model not recognized in model.get_par_3d()!')
        cumpar = np.sum(nparams[:im])
        imodel.append(range(cumpar, cumpar + nparams[im]))

    # Turn into 1D arrays (MC3 likes them this way)
    modeltype = np.array(modeltype)
    imodel    = np.array(imodel, dtype=object)
    allparams = np.array([i for item in allparams for i in item])
    allpstep  = np.array([i for item in allpstep  for i in item])
    allpmin   = np.array([i for item in allpmin   for i in item])
    allpmax   = np.array([i for item in allpmax   for i in item])
    allpnames = np.array([i for item in allpnames for i in item])
        
    return (allparams, allpstep, allpmin, allpmax, allpnames, nparams,
            modeltype, imodel)
        
    
