#! /usr/bin/env python3

# General imports
import os
import sys
import mc3
import pickle
import starry
import shutil
import subprocess
import progressbar
import numpy as np
import matplotlib.pyplot as plt

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
import mkcfg
import utils
import constants   as c
import fitclass    as fc
import taurexclass as trc

starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)

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
    print(fit.wlmid)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star, planet, system = utils.initsystem(fit, 1)

    print("Computing planet and star positions at observation times.")
    fit.x, fit.y, fit.z = [a.eval() for a in system.position(fit.t)]

    print("Calculating uniform-map planet and star fluxes.")
    fit.sflux, fit.pflux_y00 = [a.eval() for a in  \
                                system.flux(fit.t, total=False)]

    print("Calculating minimum and maximum observed longitudes.")
    fit.minvislon, fit.maxvislon = utils.vislon(planet, fit)
    print("Minimum Longitude: {:6.2f}".format(fit.minvislon))
    print("Maximum Longitude: {:6.2f}".format(fit.maxvislon))

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

    # Indices of visible cells (only considers longitudes)
    ivis = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                    (fit.lon - fit.dlon / 2. < fit.maxvislon))
    fit.ivislat, fit.ivislon = ivis
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    # List of 2d map fits
    fit.maps = []

    print("Optimizing 2D maps.")
    for i in range(len(fit.wlmid)):
        print("{:.2f} um".format(fit.wlmid[i]))
        fit.maps.append(fc.Map())

        m = fit.maps[i]
        m.ncurves = cfg.twod.ncurves[i]
        m.lmax    = cfg.twod.lmax[i]
        m.wlmid = fit.wlmid[i]

        # Where to put wl-specific outputs
        m.subdir = 'filt{}'.format(i+1)
        if not os.path.isdir(os.path.join(cfg.outdir, m.subdir)):
            os.mkdir(os.path.join(cfg.outdir, m.subdir))

        # New planet object with updated lmax
        star, planet, system = utils.initsystem(fit, m.lmax)

        print("Running PCA to determine eigencurves.")
        m.eigeny, m.evalues, m.evectors, m.ecurves, m.lcs = \
        eigen.mkcurves(system, fit.t, m.lmax, fit.pflux_y00,
                       ncurves=m.ncurves, method=cfg.twod.pca)

        print("Calculating intensities of visible grid cells of each eigenmap.")
        m.intens, m.vislat, m.vislon = eigen.intensities(planet, fit, m)

        # Set up for MCMC
        if cfg.twod.posflux:
            intens = m.intens
        else:
            intens = None
        
        indparams = (m.ecurves, fit.t, fit.pflux_y00, fit.sflux,
                     m.ncurves, m.intens)

        npar = m.ncurves + 2

        params = np.zeros(npar)
        params[m.ncurves] = 0.001
        pstep  = np.ones(npar) *  0.01
        pmin   = np.ones(npar) * -1.0
        pmax   = np.ones(npar) *  1.0

        mc3data = fit.flux[i]
        mc3unc  = fit.ferr[i]
        mc3npz = os.path.join(cfg.outdir,
                              '2dmcmc-{:.2f}um.npz'.format(fit.wlmid[i]))


        mc3out = mc3.sample(data=mc3data, uncert=mc3unc,
                            func=model.fit_2d, nsamples=cfg.twod.nsamples,
                            burnin=cfg.twod.burnin, ncpu=cfg.twod.ncpu,
                            sampler='snooker', savefile=mc3npz,
                            params=params, indparams=indparams,
                            pstep=pstep, leastsq=cfg.twod.leastsq,
                            plots=cfg.twod.plots, pmin=pmin, pmax=pmax,
                            thinning=10)

        # MC3 doesn't clear its plots >:(
        plt.close('all')

        m.bestfit = mc3out['best_model']
        m.bestp   = mc3out['bestp']
        m.stdp    = mc3out['stdp']
        m.chisq   = mc3out['best_chisq']
        m.post    = mc3out['posterior']
        m.zmask   = mc3out['zmask']

        m.nfreep = np.sum(pstep > 0)
        m.ndata  = mc3data.size

        m.redchisq = m.chisq / \
            (m.ndata - m.nfreep)
        m.bic      = m.chisq + \
            m.nfreep * np.log(m.ndata)

        print("Chisq:         {}".format(m.chisq))
        print("Reduced Chisq: {}".format(m.redchisq))
        print("BIC:           {}".format(m.bic))

        print("Calculating hotspot latitude and longitude.")
        hs = utils.hotspotloc_driver(fit, m)
        m.hslocbest  = hs[0]
        m.hslocstd   = hs[1]
        m.hslocpost  = hs[2]
        m.hsloctserr = hs[3]

        msg = "Hotspot Longitude: {:.2f} +{:.2f} {:.2f}"
        print(msg.format(m.hslocbest[1],
                         m.hsloctserr[1][0],
                         m.hsloctserr[1][1]))

        print("Calculating temperature map uncertainties.")
        m.fmappost, m.tmappost = utils.tmappost(fit, m)
        m.tmapunc = np.std(m.tmappost, axis=0)
        m.fmapunc = np.std(m.fmappost, axis=0)

    # Useful prints
    fit.totchisq2d    = np.sum([m.chisq for m in fit.maps])
    fit.totredchisq2d = fit.totchisq2d / \
        (np.sum([(m.ndata - m.nfreep) for m in fit.maps]))
    fit.totbic2d      = np.sum([m.bic for m in fit.maps])
    print("Total Chisq:         {}".format(fit.totchisq2d))
    print("Total Reduced Chisq: {}".format(fit.totredchisq2d))
    print("Total BIC:           {}".format(fit.totbic2d))
        
    # Save stellar correction terms (we need them later)
    fit.scorr = np.zeros(len(fit.wlmid))
    for i in range(len(fit.wlmid)):
        fit.scorr[i] = fit.maps[i].bestp[fit.maps[i].ncurves+1]

    print("Calculating planet visibility with time.")
    pbar = progressbar.ProgressBar(max_value=len(fit.t))
    nt = len(fit.t)
    fit.vis = np.zeros((nt, cfg.twod.nlat, cfg.twod.nlon))
    for it in range(len(fit.t)):
        fit.vis[it] = utils.visibility(fit.t[it],
                                       np.deg2rad(fit.lat),
                                       np.deg2rad(fit.lon),
                                       np.deg2rad(fit.dlat),
                                       np.deg2rad(fit.dlon),
                                       np.deg2rad(180.),
                                       cfg.planet.prot,
                                       cfg.planet.t0,
                                       cfg.planet.r,
                                       cfg.star.r,
                                       fit.x[:,it], fit.y[:,it])
        pbar.update(it+1)

    print("Checking for negative fluxes in visible cells:")
    for j in range(len(fit.wlmid)):
        print("  Wl: {:.2f} um".format(fit.wlmid[j]))
        m = fit.maps[j]
        for i in range(m.intens.shape[1]):
            check = np.sum(m.intens[:,i] *
                           m.bestp[:m.ncurves]) + \
                           m.bestp[m.ncurves] / np.pi
            if check <= 0.0:
                msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                print(msg.format(fit.vislat[i], fit.vislon[i], check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    for j in range(len(fit.wlmid)):
        star, planet, system = utils.initsystem(fit, fit.maps[j].lmax)
        fmap, tmap = eigen.mkmaps(planet, fit.maps[j].eigeny,
                                  fit.maps[j].bestp, fit.maps[j].ncurves,
                                  fit.wlmid[j], cfg.star.r, cfg.planet.r,
                                  cfg.star.t, fit.lat, fit.lon)
        fit.maps[j].fmap = fmap
        fit.maps[j].tmap = tmap

    print("Temperature ranges of maps:")
    for i in range(len(fit.wlmid)):
        print("  {:.2f} um:".format(fit.wlmid[i]))
        tmax = np.max(fit.maps[i].tmap[~np.isnan(fit.maps[i].tmap)])
        tmin = np.min(fit.maps[i].tmap[~np.isnan(fit.maps[i].tmap)])
        print("    Max: {:.2f} K".format(tmax))
        print("    Min: {:.2f} K".format(tmin))
        print("    Negative: {:f}".format(np.sum(np.isnan(fit.maps[i].tmap))))

    # Make a single array of tmaps for convenience
    fit.tmaps = np.array([m.tmap for m in fit.maps])
    fit.fmaps = np.array([m.fmap for m in fit.maps])

    if cfg.twod.plots:
        print("Making plots.")
        for m in fit.maps:
            outdir = os.path.join(cfg.outdir, m.subdir)
            plots.emaps(planet, m.eigeny, outdir, proj='ortho')
            plots.emaps(planet, m.eigeny, outdir, proj='rect')
            plots.emaps(planet, m.eigeny, outdir, proj='moll')
            plots.lightcurves(fit.t, m.lcs, outdir)
            plots.eigencurves(fit.t, m.ecurves, outdir,
                              ncurves=m.ncurves)
            plots.ecurvepower(m.evalues, outdir)
            
        plots.pltmaps(fit)
        plots.bestfit(fit)
        plots.ecurveweights(fit)
        plots.hshist(fit)

    if cfg.twod.animations:
        print("Making animations.")
        plots.visanimation(fit)
        plots.fluxmapanimation(fit)

    fit.save(cfg.outdir)

def map3d(fit, system):
    cfg = fit.cfg
    print("Fitting spectrum.")
    # Handle any atmosphere setup
    if cfg.threed.atmtype == 'ggchem':
        fit.cheminfo = atm.read_GGchem(cfg.threed.atmfile)
    else:
        fit.cheminfo = None
        
    if cfg.threed.rtfunc == 'transit':
        tcfg = mkcfg.mktransit(cfile, cfg.outdir)
        rtcall = os.path.join(transitdir, 'transit', 'transit')
        opacfile = cfg.cfg.get('transit', 'opacityfile')
        if not os.path.isfile(opacfile):
            print("  Generating opacity grid.")
            subprocess.call(["{:s} -c {:s} --justOpacity".format(rtcall, tcfg)],
                            shell=True, cwd=cfg.outdir)
        else:
            print("  Copying opacity grid: {}".format(opacfile))
            try:
                shutil.copy2(opacfile, os.path.join(cfg.outdir,
                                                    os.path.basename(opacfile)))
            except shutil.SameFileError:
                print("  Files match. Skipping.")
                pass
        subprocess.call(["{:s} -c {:s}".format(rtcall, tcfg)],
                        shell=True, cwd=cfg.outdir)

        wl, flux = np.loadtxt(os.path.join(cfg.outdir,
                                           cfg.cfg.get('transit', 'outspec')),
                              unpack=True)

    elif cfg.threed.rtfunc == 'taurex':
        fit.wngrid = np.arange(cfg.cfg.getfloat('taurex', 'wnlow'),
                               cfg.cfg.getfloat('taurex', 'wnhigh'),
                               cfg.cfg.getfloat('taurex', 'wndelt'))

        # Note: must do these things in the right order
        taurex.cache.OpacityCache().clear_cache()
        taurex.cache.OpacityCache().set_opacity_path(cfg.cfg.get('taurex',
                                                                 'csxdir'))
        taurex.cache.CIACache().set_cia_path(cfg.cfg.get('taurex',
                                                         'ciadir'))

        indparams = [fit]

        # Get sensible defaults
        params, pstep, pmin, pmax, pnames = model.get_par(fit)

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

        mc3npz = os.path.join(cfg.outdir, '3dmcmc.npz')
        

        # Build data and uncert arrays for mc3
        mc3data   = fit.flux.flatten()
        mc3uncert = fit.ferr.flatten()
        if cfg.threed.fitcf:
            ncfpar = fit.ivislat.size * len(cfg.twod.filtfiles)
            # Here we use 0s and 1s for the cf data and uncs, then
            # have the model return a value equal to the number
            # of sigma away from the cf peak, so MC3 computes the
            # correct chisq contribution from each cf
            cfdata = np.zeros(ncfpar)
            cfunc  = np.ones( ncfpar)
            mc3data   = np.concatenate((mc3data,   cfdata))
            mc3uncert = np.concatenate((mc3uncert, cfunc))

        out = mc3.sample(data=mc3data, uncert=mc3uncert,
                         func=model.mcmc_wrapper,
                         nsamples=cfg.threed.nsamples,
                         burnin=cfg.threed.burnin,
                         ncpu=cfg.threed.ncpu, sampler='snooker',
                         savefile=mc3npz, params=params,
                         indparams=indparams, pstep=pstep, pmin=pmin,
                         pmax=pmax, pnames=pnames, leastsq=None,
                         grbreak=cfg.threed.grbreak,
                         plots=cfg.threed.plots)

    fit.specbestp  = out['bestp']
    fit.chisq3d    = out['best_chisq']
    fit.redchisq3d = out['red_chisq']
    fit.bic3d      = out['BIC']

    # Put fixed params in the posterior so it's a consistent size
    fit.posterior3d = out['posterior'][out['zmask']]
    niter, nfree = fit.posterior3d.shape
    for i in range(len(params)):
        if pstep[i] == 0:
            fit.posterior3d = np.insert(fit.posterior3d, i,
                                        np.ones(niter) * params[i], axis=1)

    nfilt = len(cfg.twod.filtfiles)
    nt    = len(fit.t)

    print("Calculating best fit.")
    specout = model.specgrid(fit.specbestp, fit)
    fit.fluxgrid    = specout[0]
    fit.besttgrid   = specout[1]
    fit.taugrid     = specout[2]
    fit.p           = specout[3]
    fit.modelwngrid = specout[4]
    fit.pmaps       = specout[5]
    
    fit.specbestmodel = model.sysflux(fit.specbestp, fit)[0]
    fit.specbestmodel = fit.specbestmodel.reshape((nfilt, nt))

    print("Calculating contribution functions.")
    fit.cf = cf.contribution_filters(fit.besttgrid, fit.modelwngrid,
                                     fit.taugrid, fit.p, fit.filtwn,
                                     fit.filttrans)

    if cfg.threed.plots:
        plots.bestfitlcsspec(fit)
        plots.bestfittgrid(fit)
        plots.tau(fit)
        plots.pmaps3d(fit)
        plots.tgrid_unc(fit)
        plots.cf_by_location(fit)
        plots.cf_by_filter(fit)
        plots.cf_slice(fit)

    if cfg.threed.animations:
        plots.pmaps3d(fit, animate=True)
    
    fit.save(cfg.outdir)
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR: Call structure is run.py <mode> <configuration file>.")
        sys.exit()
    else:
        mode  = sys.argv[1]
        cfile = sys.argv[2]

    if mode in ['2d', '2D']:
        map2d(cfile)
    elif mode in ['3d', '3D']:
        # Read config to find location of output, load output,
        # then read config again to get any changes from 2d run.
        fit = fc.Fit()
        fit.read_config(cfile)
        fit = fc.load(outdir=fit.cfg.outdir)
        fit.read_config(cfile)
        # 3D mapping doesn't care about the degree of harmonics, so
        # just use 1
        star, planet, system = utils.initsystem(fit, 1)
        map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
        
    
        

    

    
