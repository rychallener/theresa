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
def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens, pindex,
           baselines, tlocs):
    """
    Basic 2D fitting routine for a single wavelength.

    Arguments
    ---------
    params: 1D float array
        Model parameters, including the map parameters and
        ramp (baseline) parameters.

    ecurves: 2D float array
        Eigencurves that are used as the fitting basis for
        the planet map.

    t: 1D float array
        ALL the times associated with this planet map. If the
        map is being fit to multiple observations, this is
        a concatenated array of those times.

    y00: 1D float array
        The light curve contribution of the uniform map component.
        Same size as t.

    sflux: 1D float array
        The light curve contribution of the star (generally,
        1 everywhere). Same size as t.

    ncurves: Int
        The number of eigencurves to use in the fit.

    intens: 2D float array
        Precomputed eigenmap intensity, of size
        (ncurves x nlocs), where nlocs is the number of locations
        where the intensity has been precomputed. This array
        is used to determine if a fit has negative intensities
        on the map, and thus can be rejected. If intens is None,
        the model will not check for negative intensities.

    pindex: 2D boolean array
        Indices used to divide params between the models. E.g.,
        params[pindex[0]] pulls out the map parameters,
        params[pindex[1]] pulls out the ramp parameters for the
        first visit, etc.

    baselines: tuple of strings
        Ramp models to use for each visit.

    tlocs: list of 1D float arrays
        Local time (relative to start of visit) for each visit.
        Used for ramp model evaluation.
    """
    mparams = params[pindex[0]]
    
    # Check for negative intensities
    if intens is not None:
        nloc = intens.shape[1]
        totint = np.zeros(nloc)
        for j in range(nloc):
            # Weighted eigenmap intensity
            totint[j] = np.sum(intens[:,j] * mparams[:ncurves])
            # Contribution from uniform map
            totint[j] += mparams[ncurves] / np.pi
        if np.any(totint <= 0):
            f = np.ones(len(t)) * np.min(totint)
            return f

    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * mparams[i]
   
    f += params[ncurves] * y00

    f += params[ncurves+1]

    # Renormalize (e.g., stellar variability between visits)
    istart = 0
    for tloc, norm in zip(tlocs, params[pindex[1]]):
        f[istart:istart + len(tloc)] *= norm
        istart += len(tloc)
        
    f += sflux
    
    # Apply ramps
    allramp = np.zeros(len(t))
    istart = 0
    for bl, tloc, ipar in zip(baselines, tlocs, pindex[2:]):
        rparams = params[ipar]
        if bl == 'none':
            ramp = np.ones(len(tloc))
        elif bl == 'linear':
            ramp = rparams[0] + rparams[1] * tloc
        elif bl == 'quadratic':
            ramp = rparams[0] +  rparams[1] * (tloc - rparams[3])**2 + \
                rparams[2] * (tloc - rparams[3])
        elif bl == 'sinusoidal':
            ramp = rparams[0] + rparams[1] * np.sin(
                2 * np.pi * tloc / rparams[2] - rparams[3])
        elif bl == 'exponential':
            ramp = rparams[0] + rparams[1] * np.exp((-rparams[2] * tloc) + rparams[3])
        elif bl == 'linexp':
            ramp = rparams[0] + rparams[1] * tloc + rparams[2] * \
                np.exp((1/rparams[3]) * -tloc)

        allramp[istart:istart + len(tloc)] += ramp
        istart += len(tloc)

    f *= allramp
    
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
   
    nlat, nlon = fit.lat.shape

    ilat = fit.ivislat3d
    ilon = fit.ivislon3d

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
        
        # TauREx spends an ungodly amount of time deciding whether to
        # print debug statements or not. Let's not.  (This is a poor
        # solution -- better to handle in TauREx but oh well)
        def do_nothing(*args):
            pass
        
        AbsCon = taurex.contributions.AbsorptionContribution()
        CIACon = taurex.contributions.CIAContribution()

        AbsCon.debug = do_nothing
        CIACon.debug = do_nothing
        rtplan.debug = do_nothing

        for molname in taurex.cache.OpacityCache().opacity_dict:
            taurex.cache.OpacityCache()[molname].debug = do_nothing
        
        if 'H-' in fit.cfg.threed.mols:
            HMCon = taurex.contributions.hm.HydrogenIon()
            HMCon.debug = do_nothing
            
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
                if 'H-' in fit.cfg.threed.mols:
                    if spec[k] == 'H':
                        gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                        rtchem.addGas(gas)
                    elif spec[k] == 'e-':
                        gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                        rtchem.addGas(gas)
            rt = trc.EmissionModel3D(
                planet=rtplan,
                star=rtstar,
                pressure_profile=rtp,
                temperature_profile=rtt,
                chemistry=rtchem,
                nlayers=cfg.threed.nlayers,
                taulimit=cfg.threed.taulimit)
            rt.add_contribution(AbsCon)
            rt.add_contribution(CIACon)
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
                rt.add_contribution(HMCon)
                    

            rt.build()

            # If we have negative temperatures, don't run the model
            # (it will fail). Return a bad fit instead. 
            if negativeT:
                fluxgrid = -1 * np.ones((nlat, nlon,
                                         len(rt.nativeWavenumberGrid)))
                return fluxgrid, rt.nativeWavenumberGrid

            wn, flux, tau, ex = rt.model(wngrid=fit.wngrid)

            # Check for very high optical depths at the top of the atmosphere
            # and print a warning
            if np.any(tau[-1] > cfg.threed.taulimit):
                print("WARNING: taulimit reached at top of the atmosphere! "
                      "Increase taulimit or decrease minimum pressure.")

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

    nlat, nlon = fit.lat.shape
    nmaps = len(fit.maps)

    # Integrate to filters
    intfluxgrid = np.zeros((nlat, nlon, nmaps))

    for i in range(nlat):
        for j in range(nlon):
            for im, m in enumerate(fit.maps):
                intfluxgrid[i,j,im] = utils.specint(wn, fluxgrid[i,j],
                                                    [m.filtwn],
                                                    [m.filttrans])

    fluxvtime = []

    # Account for vis and sum over grid cells
    for im, m in enumerate(fit.maps):
        nt = len(m.dataset.t)
        tempfvt = np.zeros(nt)
        for it in range(nt):
            tempfvt[it] = np.sum(intfluxgrid[:,:,im] * m.dataset.vis[it])
        fluxvtime.append(tempfvt)

    # There is a very small memory leak somewhere, but this seems to
    # fix it. Not an elegant solution, but ¯\_(ツ)_/¯
    gc.collect()

    return fluxvtime, tgrid, taugrid, p, wn, pmaps

def sysflux(params, fit):
    # Calculate Fp/Fs
    fpfs, tgrid, taugrid, p, wn, pmaps = specvtime(params, fit)
    nmaps = len(fpfs)                         
    systemflux = []
    # Account for stellar correction
    # Transform fp/fs -> fp/(fs + corr) -> (fp + fs + corr)/(fs + corr)
    for i, m in enumerate(fit.maps):
        fpfscorr = fpfs[i] * m.dataset.sflux / (m.dataset.sflux + fit.scorr[i])
        systemflux.append(fpfscorr + 1)
    
    return systemflux, tgrid, taugrid, p, wn, pmaps

def mcmc_wrapper(params, fit):
    tic = time.time()
    systemflux, tgrid, taugrid, p, wn, pmaps = sysflux(params, fit)

    # Flatten
    systemflux = np.concatenate(systemflux).ravel()

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
    allfiltwn    = [m.filtwn for m in fit.maps]
    allfilttrans = [m.filttrans for m in fit.maps]
    cfs = cf.contribution_filters(tgrid, wn, taugrid, p, allfiltwn,
                                  allfilttrans)

    # Where the maps "should" be
    # Find the roots of the derivative of a spline fit to
    # the contribution functions, then calculate some sort
    # of goodness of fit
    nlev, nlat, nlon = tgrid.shape
    nmaps = len(fit.maps)
    ncf = np.sum([d.ivislat.size * len(d.wlmid) for d in fit.datasets])
    cfsigdiff = np.zeros(ncf)
    logp = np.log10(p)
    order = np.argsort(logp)

    # Where to interpolate later
    xpdf = np.linspace(np.amin(logp),
                       np.amax(logp),
                       10*len(logp))
    count = 0
    for k, m in enumerate(fit.maps):
        for i, j in zip(m.dataset.ivislat, m.dataset.ivislon):
            # Check for 0 contribution and handle to avoid errors
            if np.all(cfs[i,j,order,k] == 0.0):
                print("Contribution function zero for all layers at "
                      "lat = {:.2f}, lon = {:.2f}. Likely a numerical "
                      "issue with high opacity at low pressures.".format(
                          fit.lat[i,j], fit.lon[i,j]))
                cfsigdiff += -np.inf
                return cfsigdiff
            
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

def get_par_2d(fit, d, ln):
    '''
    Returns sensible parameter settings for each 2D model
    '''
    cfg = fit.cfg
    
    # Necessary parameters
    nmappar = ln.ncurves + 2

    params = np.zeros(nmappar)
    params[ln.ncurves] = 0.001
    
    pstep = np.ones(nmappar) *  0.01
    pmin  = np.ones(nmappar) * -1.0
    pmax  = np.ones(nmappar) *  1.0

    pstep[ln.ncurves+1] = 0.0

    pnames   = []
    texnames = []
    for j in range(ln.ncurves):
        pnames.append("C{}".format(j+1))
        texnames.append("$C_{{{}}}$".format(j+1))

    pnames.append("C0")
    texnames.append("$C_0$")

    pnames.append("scorr")
    texnames.append("$s_{corr}$")

    # Renormalize paramters
    nnormpar = len(d.visits)
    params   = np.concatenate((params,   np.repeat(1.0,  nnormpar)))
    pmin     = np.concatenate((pmin,     np.repeat(0.8,  nnormpar)))
    pmax     = np.concatenate((pmax,     np.repeat(1.2,  nnormpar)))
    pnames   = np.concatenate((pnames,   ['N{}'.format(i) for i in range(1, nnormpar+1)]))
    texnames = np.concatenate((texnames, ['$N_{}$'.format(i) for i in range(1, nnormpar+1)]))
    for v in d.visits:
        # Free parameter for renormalized visits,
        # fixed to 1.0 for non-remornalized visits.
        if v.renormalize:
            pstep = np.concatenate((pstep, (0.01,)))
        else:
            pstep = np.concatenate((pstep, (0.0,)))

    nramppar = []

    # Parse baseline models
    for v in d.visits:
        if v.baseline == 'none':
            npar = 0
        elif v.baseline == 'linear':
            params   = np.concatenate((params,   (1.0, 0.0,)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001,)))
            pmin     = np.concatenate((pmin,     (0.8, -np.inf,)))
            pmax     = np.concatenate((pmax,     (1.2, np.inf,)))
            pnames   = np.concatenate((pnames,   ('b', 'm',)))
            texnames = np.concatenate((texnames, ('$b$', '$m$',)))
            npar = 1
        elif v.baseline == 'quadratic':
            params   = np.concatenate((params,   (1.0, 0.0,  0.0,   0.0)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01,  0.0)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0,  -1.0, -np.inf)))
            pmax     = np.concatenate((pmax,     (1.2, 1.0,   1.0,  np.inf)))
            pnames   = np.concatenate((pnames,   ('r0', 'r1',  'r2', 't0')))
            texnames = np.concatenate((texnames, ('r_0', '$r_1$', '$r_2$', '$t_0$')))
            npar = 3
        elif v.baseline == 'sinusoidal':
            params   = np.concatenate((params,   (1.0, -3.6e-5, 0.0885, 2.507)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001,    0.1)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0,  0.05, -np.pi)))
            pmax     = np.concatenate((pmax,     (1.2, 1.0,  0.15,  np.pi)))
            pnames   = np.concatenate((pnames,   ('b', 'Amp.', 'Period', 'Phase')))
            texnames = np.concatenate((texnames, ('$b$', 'Amp.', 'Period', 'Phase')))
            npar = 3
        elif v.baseline == 'exponential':    
            params   = np.concatenate((params,   (1.0, 0.00001, 0.00001, 0.00001)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01,    0.01)))
            pmin     = np.concatenate((pmin,     (0.8, -5,  -5, -5)))
            pmax     = np.concatenate((pmax,     (1.2, 30, 30,  30))) 
            pnames   = np.concatenate((pnames,   ('r0', 'r1', 'r2', 'r3'))) 
            texnames = np.concatenate((texnames, ('$r_0$', '$r_1$', '$r_2$', '$r_3$')))
            npar = 3
        elif v.baseline == 'linexp':
            params   = np.concatenate((params,   (1.0, -0.00219881,0.00010304,0.01629347)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001, 0.001)))
            pmin     = np.concatenate((pmin,     (0.8, -1, -0.01, 0.0)))
            pmax     = np.concatenate((pmax,     (1.2, 1, 0.01, 0.2)))
            pnames   = np.concatenate((pnames,   ('b', 'm', 'A', 'tau')))
            texnames = np.concatenate((texnames, ('$b$', '$m$', '$A$', '$\\tau$')))
            npar = 4
        else:
            print("Unrecognized baseline model.")
            sys.exit()

        nramppar.append(npar)

    npar = np.concatenate(([nmappar], [nnormpar], nramppar))
    totpar = np.sum(npar)
    cumpar = np.cumsum(npar)
    nmodel = 2 + len(d.visits)

    pindex = np.zeros((nmodel, totpar), dtype=bool)

    istart = 0
    for i in range(nmodel):
        where = np.where((np.arange(totpar) >= istart) &
                         (np.arange(totpar) <  cumpar[i]))
        pindex[i][where] = True
        istart += npar[i]

    return params, pstep, pmin, pmax, pnames, texnames, pindex

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
            pmin   = [ 150.]
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
        
    
