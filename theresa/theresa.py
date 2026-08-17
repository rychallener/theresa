#! /usr/bin/env python3

# General imports
import os
import sys
import mc3
import jax
import pickle
import shutil
import subprocess
import progressbar
import numpy as np
import mpi4py
import mpi4py.MPI
import matplotlib.pyplot as plt
import pymultinest as pym
import jaxoplanet.starry.light_curves as light_curves

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

# Taurex is a bit...talkative
import taurex.log
taurex.log.disableLogging()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')
moddir     = os.path.join(libdir,  'modules')
ratedir    = os.path.join(moddir,  'rate')
transitdir = os.path.join(moddir, 'transit')

# Lib imports
sys.path.append(libdir)
import cf
import atm
import pca
import eigen
import model
import plots
import utils
import constants as c
import fitclass  as fc

# Work in double precision with JAX
jax.config.update("jax_enable_x64", True)

# MPI
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

def map2d(cfile):
    """
    One function to rule them all.
    """
    # Create the master fit object
    fit = fc.Fit()
    
    print("Reading the configuration file.")
    fit.read_config(cfile)
    cfg = fit.cfg

    print("Reading the data.")
    fit.read_data()

    print("Reading filters.")
    fit.read_filters()
    print("Filter mean wavelengths (um):")
    for d in fit.datasets:
        print(d.wlmid)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star, planet, system = utils.initsystem(fit, 1)

    # This is basically doing the angular size for each entry in the
    # array (hence the dlat)
    print("Calculating latitude and longitude of planetary grid.")
    fit.dlat = 180. / cfg.twod.nlat
    fit.dlon = 360. / cfg.twod.nlon
    fit.lat, fit.lon = np.meshgrid(np.linspace(-90  + fit.dlat / 2.,
                                                90  - fit.dlat / 2.,
                                               cfg.twod.nlat, endpoint=True),
                                   np.linspace(-180 + fit.dlon / 2.,
                                                180 - fit.dlon / 2.,
                                               cfg.twod.nlon, endpoint=True),
                                   indexing='ij')
    fit.dlatgrid, fit.dlongrid = np.meshgrid(np.ones(cfg.twod.nlat) * fit.dlat,
                                             np.ones(cfg.twod.nlon) * fit.dlon,
                                             indexing='ij')

    for d in fit.datasets:
        print("Precomputing - {}".format(d.name))

        print("Computing planet and star positions at observation times.")
        px, py, pz = system.bodies[0].position(d.t)
        sx, sy, sz = system.central_position(d.t)
        d.x = np.vstack([sx, px])
        d.y = np.vstack([sy, py])
        d.z = np.vstack([sz, pz])

        print("Calculating uniform-map planet and star fluxes.")
        lcfunc = light_curves.light_curve(system)
        d.sflux, d.pflux_y00 = lcfunc(d.t).T
        # Convert to numpy arrays for numba compatibility
        d.sflux = np.array(d.sflux)
        d.pflux_y00 = np.array(d.pflux_y00)

        print("Calculating minimum and maximum observed longitudes.")
        d.minvislon, d.maxvislon = utils.vislon(system, d)
        print("Minimum Longitude: {:6.2f}".format(d.minvislon))
        print("Maximum Longitude: {:6.2f}".format(d.maxvislon))

        # Indices of visible cells (only considers longitudes)
        ivis = np.where((fit.lon + fit.dlon / 2. > d.minvislon) &
                        (fit.lon - fit.dlon / 2. < d.maxvislon))
        d.ivislat, d.ivislon = ivis
    
    if not os.path.isdir(cfg.twod.outdir):
        os.mkdir(cfg.twod.outdir)

    print("Optimizing 2D maps.")
    for d in fit.datasets:
        d.maps = []
        for i in range(len(d.wlmid)):
            print("{:.2f} um".format(d.wlmid[i]))
            m = fc.Map()
            
            d.maps.append(m)
            
            m.wlmid     = d.wlmid[i]
            m.filtwl    = d.filtwl[i]
            m.filtwn    = d.filtwn[i]
            m.filttrans = d.filttrans[i]
            m.flux      = d.flux[i]
            m.ferr      = d.ferr[i]

            # Where to put wl-specific outputs
            m.subdir = '{}-filt{}'.format(d.name, i+1)
            if not os.path.isdir(os.path.join(cfg.twod.outdir, m.subdir)):
                os.mkdir(os.path.join(cfg.twod.outdir, m.subdir))

            minbic = np.inf

            for l in range(1, cfg.twod.lmax+1):
                for n in range(0, cfg.twod.ncurves+1):
                    # Skip cases where n is higher than the number of
                    # available eigencurves, which is (l+1)**2, minus
                    # the uniform (l=0) case, since that's included by
                    # default
                    if n > (l+1)**2 - 1:
                        continue

                    # Also let's only do the n=0 case once, since
                    # it's exactly the same fit for every lmax.
                    # Link the LN objects for looping simplicity later
                    if l > 1 and n==0:
                        setattr(m, 'l{}n{}'.format(l, n), m.l1n0)
                        continue

                    print("Fitting lmax={}, n={}".format(l,n))
                    setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                    ln = getattr(m, 'l{}n{}'.format(l, n))

                    ln.subdir = 'l{}n{}'.format(l,n)

                    ln.wlmid = d.wlmid[i]

                    ln.ncurves = n
                    ln.lmax    = l

                    # New planet object with updated lmax
                    star, planet, system = utils.initsystem(fit, ln.lmax)

                    print("Running PCA to determine eigencurves.")
                    ncomp = ln.ncurves
                    if ln.ncurves == 0:
                        ncomp = None
                        
                    ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        eigen.mkcurves(fit, d, ln.lmax, ncurves=ncomp,
                                       method=cfg.twod.pca,
                                       orbcheck=cfg.twod.orbcheck,
                                       sigorb=cfg.twod.sigorb)

                    print("Calculating intensities of visible grid cells of each eigenmap.")
                    ln.intens, ln.vislat, ln.vislon = \
                        eigen.intensities(fit, d, ln)

                    # Set up for MCMC
                    if cfg.twod.posflux:
                        intens = ln.intens
                    else:
                        intens = None

                    params, pstep, pmin, pmax, pnames, texnames, pindex = \
                        model.get_par_2d(fit, d, ln)

                    baselines = tuple(v.baseline for v in d.visits)

                    tlocs = tuple(v.tloc for v in d.visits)

                    dvecs = tuple(v.dvec for v in d.visits)
                        
                    indparams = (ln.ecurves, d.t, d.pflux_y00, d.sflux,
                                 ln.ncurves, intens, pindex,
                                 baselines, tlocs, dvecs)

                    # Better initial guess if possible
                    if hasattr(m, "l{}n{}".format(l,n-1)):
                        params = getattr(m, "l{}n{}".format(l,n-1)).bestp
                        params = np.insert(params, n-1, 0.0)

                    mc3data = d.flux[i]
                    mc3unc  = d.ferr[i]
                    mc3npz = os.path.join(cfg.twod.outdir,
                                          m.subdir,
                                          ln.subdir,
                                          '2dmcmc-l{}n{}-{:.2f}um.npz'.format(
                                              l,
                                              n,
                                              d.wlmid[i]))


                    # Make sure we don't use too much RAM
                    thinning = int(np.max((10, cfg.twod.nsamples // 1e5)))
                    mc3out = mc3.sample(data=mc3data, uncert=mc3unc,
                                        func=model.fit_2d,
                                        nsamples=cfg.twod.nsamples,
                                        burnin=cfg.twod.burnin,
                                        ncpu=cfg.twod.ncpu, sampler='snooker',
                                        savefile=mc3npz, params=params,
                                        indparams=indparams, pstep=pstep,
                                        leastsq=cfg.twod.leastsq,
                                        plots=cfg.twod.plots, pmin=pmin,
                                        pmax=pmax, pnames=pnames,
                                        texnames=texnames, thinning=thinning,
                                        fgamma=cfg.twod.fgamma,
                                        grbreak=1.01)

                    # MC3 doesn't clear its plots >:(
                    plt.close('all')

                    ln.bestfit = mc3out['best_model']
                    ln.bestp   = mc3out['bestp']
                    ln.stdp    = mc3out['stdp']
                    ln.chisq   = mc3out['best_chisq']
                    ln.post    = mc3out['posterior']
                    ln.zmask   = mc3out['zmask']

                    # Isolate systematics models (used later in 3d mapping)
                    # Do this by calculating the best-fitting model without
                    # systematics and dividing it out of the best-fitting
                    # model
                    nobaselines = tuple('none' for v in d.visits)
                    nodvecs     = \
                        tuple(np.zeros((len(v.t), 1),
                                       dtype=float).T for v in d.visits)
                    
                    nosysmodel = model.fit_2d(ln.bestp, ln.ecurves,
                                              d.t, d.pflux_y00,
                                              d.sflux, ln.ncurves,
                                              intens, pindex,
                                              nobaselines, tlocs,
                                              nodvecs)

                    ln.systematics = ln.bestfit / nosysmodel

                    niter, nfree = ln.post.shape
                    nparams = len(params)
                    for ip in range(nparams):
                        if pstep[ip] == 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * params[ip],
                                axis=1)
                        if pstep[ip] < 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * ln.bestp[-int(pstep[ip])],
                                axis=1)

                    ln.nfreep = np.sum(pstep > 0)
                    ln.ndata  = mc3data.size

                    ln.redchisq = ln.chisq / \
                        (ln.ndata - ln.nfreep)
                    ln.bic      = ln.chisq + \
                        ln.nfreep * np.log(ln.ndata)

                    print("Chisq:         {}".format(ln.chisq))
                    print("Reduced Chisq: {}".format(ln.redchisq))
                    print("BIC:           {}".format(ln.bic))

                    if ln.bic < minbic:
                        minbic = ln.bic
                        m.bestln = ln

            print("Calculating hotspot latitude and longitude.")
            hs = utils.hotspotloc_driver(fit, m.bestln)
            m.hslocbest  = hs[0]
            m.hslocstd   = hs[1]
            m.hslocpost  = hs[2]
            m.hsloctserr = hs[3]

            msg = "Hotspot Longitude: {:.2f} +{:.2f} {:.2f}"
            print(msg.format(m.hslocbest[1],
                             m.hsloctserr[1][0],
                             m.hsloctserr[1][1]))

            # Populate blackbody spectra outside posterior map
            # calculation loop for speed
            # Note: numerical issues can occur below 50 K, but it's
            #       possible that the model returns a map with fluxes
            #       low enough for such cold temperatures, which
            #       could result in issues in the future.
            if fit.cfg.star.starspec == 'custom':
                # Temperatures for later interpolation
                m.trange = np.linspace(50, 5000, 10000)
                # Blackbody spectra at each temperature
                bbs = utils.blackbody_wl(m.trange, m.filtwl * 1e-6)
                # Interpolated stellar spectrum
                sspec_int = np.interp(m.filtwl, fit.starwl, fit.starflux)
                # Band-integrated stellar spectrum
                sspec_fint = np.trapezoid(m.filttrans * sspec_int,
                                          m.filtwl * 1e-6)
                rprs2 = (fit.cfg.planet.r / fit.cfg.star.r)**2
                fpfs_for_bbs = rprs2 * bbs / sspec_int
                m.fpfs_for_interp = np.trapezoid(
                    fpfs_for_bbs * m.filttrans * sspec_int,
                    m.filtwl * 1e-6, axis=1) / sspec_fint
                
            else:
                m.trange          = None
                m.fpfs_for_interp = None

            print("Calculating flux and temperature map uncertainties.")
            m.fmappost, m.tmappost = utils.tmappost(fit, m, m.bestln)
            m.tmapunc = np.std(m.tmappost, axis=0)
            m.fmapunc = np.std(m.fmappost, axis=0)

    print("Optimum lmax and ncurves:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  {:.2f} um: lmax={}, ncurves={}".format(m.wlmid,
                                                            m.bestln.lmax,
                                                            m.bestln.ncurves))
        
    # Save stellar correction terms (we need them later)
    #fit.scorr = np.zeros(len(fit.maps))
    #for i in range(len(fit.maps)):
    #    fit.scorr[i] = fit.maps[i].bestln.bestp[fit.maps[i].bestln.ncurves+1]

    print("Checking for negative fluxes in visible cells:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  Wl: {:.2f} um".format(m.wlmid))
            for i in range(m.bestln.intens.shape[1]):
                check = np.sum(m.bestln.intens[:,i] *
                               m.bestln.bestp[:m.bestln.ncurves]) + \
                               m.bestln.bestp[ m.bestln.ncurves] / np.pi
                if check <= 0.0:
                    msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                    print(msg.format(m.dataset.vislat[i],
                                     m.dataset.vislon[i],
                                     check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    for d in fit.datasets:
        for m in d.maps:
            star, planet, system = utils.initsystem(fit, m.bestln.lmax)
            fmap, tmap = eigen.mkmaps(fit, m, m.bestln, m.bestln.bestp)
            m.fmap = fmap
            m.tmap = tmap

    print("Temperature ranges of maps:")
    for d in fit.datasets:
        for m in d.maps:
            print("  {:.2f} um:".format(m.wlmid))
            tmax = np.max(m.tmap[~np.isnan(m.tmap)])
            tmin = np.min(m.tmap[~np.isnan(m.tmap)])
            print("    Max: {:.2f} K".format(tmax))
            print("    Min: {:.2f} K".format(tmin))
            print("    Negative: {:f}".format(np.sum(np.isnan(m.tmap))))

    # Make a single array of tmaps for convenience
    fit.nmaps = np.sum([len(d.maps) for d in fit.datasets])
    fit.tmaps2d = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    fit.fmaps2d = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))

    imap = 0
    for d in fit.datasets:
        for m in d.maps:
            fit.tmaps2d[imap] = m.tmap
            fit.fmaps2d[imap] = m.fmap
            imap += 1

    # Save fit object before plotting in case of crashes
    fit.save(cfg.twod.outdir)

    if cfg.twod.plots:
        print("Making plots.")
        for d in fit.datasets:
            for m in d.maps:
                outdir = os.path.join(cfg.twod.outdir, m.subdir)
                # Make sure the planet has the right lmax
                star, planet, system = utils.initsystem(fit, m.bestln.lmax)
                plots.emaps(fit, m.bestln.eigeny, outdir, proj='rect')
                plots.lightcurves(d.t, m.bestln.lcs, outdir)
                plots.eigencurves(d.t, m.bestln.ecurves, outdir,
                                  ncurves=m.bestln.ncurves)
                plots.ecurvepower(m.bestln.evalues, outdir)
            
        plots.pltmaps(fit)
        plots.tmap_unc(fit)
        plots.bestfit(fit, outdir=cfg.twod.outdir)
        plots.ecurveweights(fit)
        plots.hshist(fit)
        plots.bics(fit, outdir=cfg.twod.outdir)

    # With the new grid and visibility calculation moved to the 3D
    # function, these no longer function
    if cfg.twod.animations:
        pass
        #plots.visanimation(fit, outdir=cfg.twod.outdir)
        #plots.fluxmapanimation(fit, outdir=cfg.twod.outdir)

def map3d(fit, system):
    cfg = fit.cfg
    outdir = os.path.join(cfg.threed.indir, cfg.threed.outdir)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # Handle any atmosphere setup
    if cfg.threed.atmtype == 'ggchem':
        if cfg.cfg.has_option('GGchem', 'dispolfiles'):
            dispolfiles = cfg.cfg.get('GGchem', 'dispolfiles')
        else:
            dispolfiles = None

        # TODO: this is gross. Should just allow user to specify the
        #       file in the configuration.
        defaultgrid = ((cfg.threed.nlayers == 100) &
                       (cfg.threed.ptop  == 1e-6)   &
                       (cfg.threed.pbot  == 1e2)    &
                       (cfg.threed.numt  ==   77)   &
                       (cfg.threed.tmin  ==  150)   &
                       (cfg.threed.tmax  == 4000)   &
                       (cfg.threed.numz  == 41)     &
                       (cfg.threed.zmin  == -2.0)   &
                       (cfg.threed.zmax  ==  2.0)   &
                       (cfg.threed.comin == -2.0)  &
                       (cfg.threed.comax ==  0.0)   &
                       (cfg.threed.numco == 10)    &
                       (cfg.threed.mols == ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'C2H2', 'C2H4', 'HCN', 'H2S']) &
                       (dispolfiles is None)       &
                       (cfg.threed.elem == ['H', 'He', 'C', 'N', 'O', 'S']) &
                       (cfg.threed.condensates == False))

        if defaultgrid:
            print("Loading default chemistry grid.")
            cheminfo = np.load('inputs/ggchem-default.npz')
            fit.cheminfo = (cheminfo['T'],
                            cheminfo['P'],
                            cheminfo['z'],
                            cheminfo['co'],
                            cheminfo['spec'],
                            cheminfo['abn'])
            del(cheminfo)
        else:
            print("Precomputing chemistry grid.")
            # T, P, z, spec, abn
            fit.cheminfo = atm.setup_GGchem(cfg.threed.tmin,
                                            cfg.threed.tmax,
                                            cfg.threed.numt,
                                            cfg.threed.ptop,
                                            cfg.threed.pbot,
                                            cfg.threed.nlayers,
                                            cfg.threed.zmin,
                                            cfg.threed.zmax,
                                            cfg.threed.numz,
                                            cfg.threed.comin,
                                            cfg.threed.comax,
                                            cfg.threed.numco,
                                            cfg.threed.mols,
                                            cfg.threed.cmols,
                                            condensates=cfg.threed.condensates,
                                            elements=cfg.threed.elem,
                                            dispolfiles=dispolfiles)
    else:
        fit.cheminfo = None

    print("Pre-calculating planet visibility with time.")
    # Determine highest lmax which sets how much we sample
    # the 3D grid
    vis_lmax = 0
    for d in fit.datasets:
        for m in d.maps:
            if m.bestln.lmax > vis_lmax:
                vis_lmax = m.bestln.lmax
                
    for d in fit.datasets:
        print(d.name)
        d.vis, d.lat3d, d.lon3d = utils.visibility(
            fit, d, vis_lmax)

    # These are the same for all datasets
    fit.lat3d = fit.datasets[0].lat3d
    fit.lon3d = fit.datasets[0].lon3d

    fit.ncolumn = fit.datasets[0].vis.shape[1]

    # Determine which grid cells to use
    # Figures out which grid cells have any visibility for each dataset,
    # folds them together with an "or" operation, then filters the column
    # indices down to those which are visible
    totvisbool = np.zeros(fit.ncolumn)
    for d in fit.datasets:
        visbool = np.zeros(fit.ncolumn)
        for ic in range(fit.ncolumn):
            if np.any(d.vis[ic] > 0):
                visbool[ic] = 1

        totvisbool = np.logical_or(totvisbool, visbool)
        
    fit.ivis3d = np.arange(0, fit.ncolumn)[totvisbool]

    # Make a single array of tmaps on the 3D grid
    fit.nmaps = np.sum([len(d.maps) for d in fit.datasets])
    fit.tmaps3d = np.zeros((fit.nmaps, fit.ncolumn))
    fit.fmaps3d = np.zeros((fit.nmaps, fit.ncolumn))

    print("Calculating 2D temperature maps at oversample resolution.")
    if fit.cfg.threed.nightavg:
        print("Averaging nightside temperatures (this may take a while).")
        pbar = progressbar.ProgressBar(max_value=fit.nmaps*fit.ncolumn)
        
    ncurves3d = np.max([m.bestln.ncurves for m in d.maps for d in fit.datasets])
    imap = 0
    
    for d in fit.datasets:
        for m in d.maps:
            star, planet, system = utils.initsystem(fit, m.bestln.lmax)
            ln = getattr(m, 'l{}n{}'.format(m.bestln.lmax, ncurves3d))
            fmap, tmap = eigen.mkmaps(fit, m, ln, ln.bestp,
                                      lat=fit.lat3d,
                                      lon=fit.lon3d)
            
            fit.tmaps3d[imap] = tmap
            fit.fmaps3d[imap] = fmap
            
            # Flatten out the nightside if asked for
            # This calculates a latitudinal slice of the flux map at each
            # longitude on the nightside, takes the vis-weighted average,
            # then computes a temperature. Not terribly efficient due
            # to the Mollweide grid, but oh well
            if fit.cfg.threed.nightavg:
                for il, l in enumerate(fit.lon3d):
                    if l < -90. or l > 90.:
                        templat = np.linspace(-90., 90., 100)
                        templon = np.ones(100) * l
                        fslice, tslice = eigen.mkmaps(fit, m, ln,
                                                      ln.bestp,
                                                      lat=templat,
                                                      lon=templon)

                        favg = np.mean(fslice * np.cos(np.deg2rad(templat))) \
                                       / np.mean(np.cos(np.deg2rad(templat)))

                        tavg = utils.fmap_to_tmap(
                            favg, m.wlmid, fit.cfg.planet.r,
                            fit.cfg.star.r, fit.cfg.star.t,
                            m.bestln.bestp[m.bestln.ncurves],
                            starspec=fit.cfg.star.starspec, fwl=fwl,
                            ftrans=ftrans, swl=swl, sspec=sspec)

                        fit.fmaps3d[imap][il] = favg
                        fit.tmaps3d[imap][il] = tavg

                    pbar.update(imap*fit.ncolumn+il)

            imap += 1
            

    # Make array of systematics models for correcting light curves in
    # 3d fitting
    fit.systematics3d = []
    for d in fit.datasets:
        for m in d.maps:
            ln = getattr(m, 'l{}n{}'.format(m.bestln.lmax, ncurves3d))
            fit.systematics3d.append(ln.systematics)

    print("Fitting spectrum.")
    if cfg.threed.rtfunc == 'taurex':
        # Make sure the wn range is appropriate
        wnlow  = cfg.cfg.getfloat('taurex', 'wnlow')
        wnhigh = cfg.cfg.getfloat('taurex', 'wnhigh')
        wndelt = 1.0

        for d in fit.datasets:
            for m in d.maps:
                filtwn = m.filtwn
                filttrans = m.filttrans
                nonzero = filtwn[np.where(filttrans != 0.0)]
                if not np.all((nonzero > wnlow) & (nonzero < wnhigh)):
                    print("ERROR: Wavenumber range does not cover all filters!")
                    sys.exit()
                
        fit.wngrid = np.arange(wnlow, wnhigh, wndelt)

        # Note: must do these things in the right order
        taurex.cache.OpacityCache().clear_cache()
        taurex.cache.OpacityCache().set_opacity_path(cfg.cfg.get('taurex',
                                                                 'csxdir'))
        taurex.cache.CIACache().set_cia_path(cfg.cfg.get('taurex',
                                                         'ciadir'))

        indparams = [fit]

        # Get sensible defaults
        params, pstep, pmin, pmax, pnames, nparams, modeltype, imodel = \
            model.get_par_3d(fit)

        fit.nparams3d   = nparams
        fit.modeltype3d = modeltype
        fit.imodel3d    = imodel

        # Override if specified by the user
        if hasattr(cfg.threed, 'params'):
            params = cfg.threed.params
        if hasattr(cfg.threed, 'pmin'):
            pmin   = cfg.threed.pmin
        if hasattr(cfg.threed, 'pmax'):
            pmax   = cfg.threed.pmax
        if hasattr(cfg.threed, 'pstep'):
            pstep  = cfg.threed.pstep
        if hasattr(cfg.threed, 'pnames'):
            pnames = cfg.threed.pnames

        nparams = len(params)

        mc3npz = os.path.join(outdir, '3dmcmc.npz')
        

        # Build data and uncert arrays for mc3
        mcdata   = \
            np.concatenate([m.flux for d in fit.datasets for m in d.maps])
        mcuncert = \
            np.concatenate([m.ferr for d in fit.datasets for m in d.maps])
        
        if cfg.threed.fitcf:
            ncfpar = fit.ivis3d.size * fit.nmaps
            print("ncf: " + str(ncfpar))
            # Here we use 0s and 1s for the cf data and uncs, then
            # have the model return a value equal to the number
            # of sigma away from the cf peak, so MC3 computes the
            # correct chisq contribution from each cf
            cfdata = np.zeros(ncfpar)
            cfunc  = np.ones( ncfpar)
            mcdata   = np.concatenate((mc3data,   cfdata))
            mcuncert = np.concatenate((mc3uncert, cfunc))

        # Avoid crashing if user tries to resume a run that never
        # happened
        if os.path.isfile(mc3npz) and cfg.threed.resume:
            resume = True
        else:
            resume = False

        # Avoid common mc3 crash if previous run was killed
        # or crashed
        if resume and cfg.threed.sampler == 'mc3':
            oldrun = np.load(mcnpz)
            oldrun = dict(oldrun)
            if not 'chisq_factor' in oldrun:
                oldrun['chisq_factor'] = 1.0
                np.savez(mc3npz, **oldrun)
                # Let's not keep an extra posterior
                # in memory
                del(oldrun)
                
        if cfg.threed.sampler == 'mc3':
            out = mc3.sample(data=mc3data, uncert=mc3uncert,
                             func=model.mcmc_wrapper,
                             nsamples=cfg.threed.nsamples,
                             burnin=cfg.threed.burnin,
                             #ncpu=cfg.threed.ncpu,
                             sampler='snooker',
                             savefile=mc3npz, params=params,
                             indparams=indparams,
                             pstep=pstep, pmin=pmin,
                             pmax=pmax, pnames=pnames,
                             leastsq=cfg.threed.leastsq,
                             grbreak=cfg.threed.grbreak,
                             fgamma=cfg.threed.fgamma,
                             plots=cfg.threed.plots,
                             resume=resume)

            fit.specbestp   = out['bestp']
            fit.chisq3d     = out['best_chisq']
            fit.redchisq3d  = out['red_chisq']
            fit.bic3d       = out['BIC']
            fit.zmask3d     = out['zmask']
            fit.zchain3d    = out['zchain']
            fit.posterior3d = out['posterior']

            # Put fixed and shared params in the posterior so it's a
            # consistent size
            niter, nfree = fit.posterior3d.shape
            for i in range(nparams):
                if pstep[i] == 0:
                    fit.posterior3d = np.insert(
                        fit.posterior3d, i,
                        np.ones(niter) * params[i],
                        axis=1)
                if pstep[i] < 0:
                    fit.posterior3d = np.insert(
                        fit.posterior3d, i,
                        np.ones(niter) * fit.specbestp[-int(pstep[i])],
                        axis=1)

            # Evaluate SPEIS, ESS, and CR error
            print("Calculating effective sample size.")
            nchains = np.max(fit.zchain3d) + 1
            fit.cspeis3d = np.zeros((nchains, nparams)) # SPEIS by chain
            fit.cess3d   = np.zeros((nchains, nparams)) # ESS by chain
            for i in range(nchains):
                where = np.where(fit.zchain3d[fit.zmask3d] == i)
                chain = fit.posterior3d[fit.zmask3d][where]
                if len(chain) == 0:
                    print('WARNING: Chain {} has no accepted iterations!'.format(i))
                else:
                    fit.cspeis3d[i], fit.cess3d[i] = utils.ess(chain)

            fit.ess3d   = np.sum(fit.cess3d, axis=0) # Overall ESS
            fit.speis3d = np.ceil(niter / fit.ess3d).astype(int) # Overall SPEIS
            fit.crsig3d = np.zeros(nparams)
            for i in range(nparams):
                fit.crsig3d[i] = utils.crsig(fit.ess3d[i])

            print("\nParameter        SPEIS     ESS   68.3% Error"
                  "\n-------------- ------- ------- -------------")
            for i in range(nparams):
                if pstep[i] == 0:
                    continue
                print(f"{pnames[i]:<14s} " +
                      f"{fit.speis3d[i]:7d} " +
                      f"{fit.ess3d[i]:7.1f} " +
                      f"{fit.crsig3d[i]:13.2e}")

            # MC3 doesn't clear its plots >:(
            plt.close('all')
        elif cfg.threed.sampler == 'multinest':
            print('Running PyMultiNest retrieval.')
            basename = os.path.join(cfg.twod.outdir, cfg.threed.outdir) + '/'
            model.pym_retrieval(fit, mcdata, mcuncert, pmin, pmax,
                                n_live_points=400,
                                outputfiles_basename=basename,
                                verbose=True,
                                resume=cfg.threed.resume,
                                importance_nested_sampling=False)

            analyzer = pym.Analyzer(nparams,
                                    outputfiles_basename=basename,
                                    verbose=False)

            fit.specbestp   = np.array(analyzer.get_best_fit()['parameters'])
            fit.loglike3d   = analyzer.get_best_fit()['log_likelihood']
            fit.posterior3d = analyzer.get_equal_weighted_posterior()[:,:-1]

            niter, nfree = fit.posterior3d.shape
          
    nmaps = fit.nmaps

    print("Calculating best fit.")
    specout = model.specgrid(fit.specbestp, fit)
    fit.fluxgrid    = specout[0]
    fit.besttgrid   = specout[1]
    fit.taugrid     = specout[2]
    fit.p           = specout[3]
    fit.modelwngrid = specout[4]
    fit.pmaps       = specout[5]
    
    fit.specbestmodel = model.sysflux(fit.specbestp, fit)[0]
    rawbestmodel = model.mcmc_wrapper(fit.specbestp, fit)

    # Calculate chisq of just light curve (ignoring cf penalty)
    lcs  = [m.flux for d in fit.datasets for m in d.maps]
    elcs = [m.ferr for d in fit.datasets for m in d.maps]
    fit.chisq3d    = np.sum(((rawbestmodel - mcdata) / mcuncert)**2)
    fit.chisq3d_lc = 0
    for i in range(len(lcs)):
        fit.chisq3d_lc += np.sum(((lcs[i] - fit.specbestmodel[i]) / elcs[i])**2)

    nlcdata = len(np.concatenate(lcs))
    fit.redchisq3d_lc = fit.chisq3d / (nlcdata - nfree)
    print("Light Curve Chisq:      {:.2f}".format(fit.chisq3d_lc))
    print("Light Curve Red. Chisq: {:.4f}".format(fit.redchisq3d_lc))
    print("CF Penalty:             {:.2f}".format(fit.chisq3d - fit.chisq3d_lc))

    allmols = np.concatenate((cfg.threed.mols, cfg.threed.cmols))
    if type(fit.cfg.threed.z) is float:
        z = fit.cfg.threed.z
    elif fit.cfg.threed.z == 'fit':
        izmodel = np.where(fit.modeltype3d == 'z')[0][0]
        istart = np.sum(fit.nparams3d[:izmodel])
        z = params[istart]
    else:
        print("Something has gone wrong.")

    if type(fit.cfg.threed.co) is float:
        co = fit.cfg.threed.co
    elif fit.cfg.threed.z == 'fit':
        icomodel = np.where(fit.modeltype3d == 'c/o')[0][0]
        istart = np.sum(fit.nparams3d[:icomodel])
        co = params[istart]
    else:
        print("Something has gone wrong.")
        
    fit.abnbest, fit.abnspec, _ = atm.atminit(fit.cfg.threed.atmtype,
                                              allmols, fit.p,
                                              fit.besttgrid, z, co,
                                              ivis=fit.ivis3d,
                                              cheminfo=fit.cheminfo)
                                           

    print("Calculating contribution functions.")
    allfiltwn    = [m.filtwn    for d in fit.datasets for m in d.maps]
    allfilttrans = [m.filttrans for d in fit.datasets for m in d.maps]
    fit.cf = cf.contribution_filters(fit.besttgrid, fit.modelwngrid,
                                     fit.taugrid, fit.p, allfiltwn,
                                     allfilttrans)

    # Save before plots, in case of crashes
    # Do not add attributes to fit after this
    if rank == 0:
        fit.save(outdir)
    
    if cfg.threed.plots and rank == 0:
        plots.bestfitlcsspec(fit, outdir=outdir)
        plots.bestfittgrid(fit, outdir=outdir)
        plots.tau(fit, outdir=outdir)
        plots.tgrid_unc(fit, outdir=outdir)
        plots.cf_by_filter(fit, outdir=outdir)
        plots.spectra(fit, outdir=outdir)
        plots.spatialsampling(fit, outdir=outdir)
        plots.abundances(fit, outdir=outdir)
        plots.photospheres(fit, outdir=outdir)
        # TODO: generalize this plot for all temperature structures
        if 'tgcm' in fit.cfg.threed.modelnames:
            plots.isobars(fit, outdir=outdir)
            plots.radadv(fit, outdir=outdir)

    # There actually aren't any of these at the moment
    if cfg.threed.animations:
        pass

        
if __name__ == "__main__":
    if rank == 0:
        print("#########################################################")
        print("  ThERESA: Three-dimensional Exoplanet Retrieval from    ")
        print("           Eclipse Spectroscopy of Atmospheres           ")
        print("  Copyright 2021-2026 Ryan C. Challener & collaborators  ")
        print("#########################################################")
    
    if len(sys.argv) < 3:
        print("ERROR: Call structure is theresa.py <mode> <configuration file>.")
        sys.exit()
    else:
        mode  = sys.argv[1]
        cfile = sys.argv[2]

    if mode in ['2d', '2D']:
        map2d(cfile)
    elif mode in ['3d', '3D']:
        # Read config to find location of output, load output,
        # then read config again to get any changes from 2d run.
        if rank == 0:
            fit = fc.Fit()
            fit.read_config(cfile)
            fit = fc.load(outdir=fit.cfg.threed.indir)
            fit.read_config(cfile)
        else:
            fit = None

        fit = comm.bcast(fit, root=0)
        # 3D mapping doesn't care about the degree of harmonics, so
        # just use 1
        star, planet, system = utils.initsystem(fit, 1)
        map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
        
    
        

    

    
