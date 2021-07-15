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
    star, planet, system = utils.initsystem(fit)

    print("Computing planet and star positions at observation times.")
    fit.x, fit.y, fit.z = [a.eval() for a in system.position(fit.t)]

    print("Calculating uniform-map planet and star fluxes.")
    fit.sflux, fit.pflux_y00 = [a.eval() for a in  \
                                system.flux(fit.t, total=False)]

    print("Running PCA to determine eigencurves.")
    fit.eigeny, fit.evalues, fit.evectors, fit.ecurves, fit.lcs = \
        eigen.mkcurves(system, fit.t, cfg.twod.lmax, fit.pflux_y00,
                       ncurves=fit.cfg.twod.ncurves, method=cfg.twod.pca)

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

    print("Calculating intensities of visible grid cells of each eigenmap.")
    fit.intens, fit.vislat, fit.vislon = eigen.intensities(planet, fit)
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    # List of 2d map fits
    fit.maps = []

    # Set up for MCMC
    if cfg.twod.posflux:
        intens = fit.intens
    else:
        intens = None
        
    indparams = (fit.ecurves, fit.t, fit.pflux_y00, fit.sflux,
                 cfg.twod.ncurves, intens)

    npar = cfg.twod.ncurves + 2

    print("Optimizing 2D maps.")
    for i in range(len(fit.wlmid)):
        print("  {:.2f} um".format(fit.wlmid[i]))
        fit.maps.append(fc.Map())

        params = np.zeros(npar)
        params[cfg.twod.ncurves] = 0.001
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

        fit.maps[i].bestfit = mc3out['best_model']
        fit.maps[i].bestp   = mc3out['bestp']
        fit.maps[i].stdp    = mc3out['stdp']
        fit.maps[i].chisq   = mc3out['best_chisq']
        fit.maps[i].post    = mc3out['posterior']
        fit.maps[i].zmask   = mc3out['zmask']

        fit.maps[i].nfreep = np.sum(pstep > 0)
        fit.maps[i].ndata  = mc3data.size

        fit.maps[i].redchisq = fit.maps[i].chisq / \
            (fit.maps[i].ndata - fit.maps[i].nfreep)
        fit.maps[i].bic      = fit.maps[i].chisq + \
            fit.maps[i].nfreep * np.log(fit.maps[i].ndata)

        print("Chisq:         {}".format(fit.maps[i].chisq))
        print("Reduced Chisq: {}".format(fit.maps[i].redchisq))
        print("BIC:           {}".format(fit.maps[i].bic))

        print("Calculating hotspot latitude and longitude.")
        fit.maps[i].hslocbest, fit.maps[i].hslocstd, fit.maps[i].hslocpost = \
            utils.hotspotloc_driver(fit, fit.maps[i])

        print("Hotspot Longitude: {} +/- {}".format(fit.maps[i].hslocbest[1],
                                                    fit.maps[i].hslocstd[1]))

        print("Calculating temperature map uncertainties.")
        fit.maps[i].wlmid = fit.wlmid[i]
        fit.maps[i].fmappost, fit.maps[i].tmappost = utils.tmappost(
            fit, fit.maps[i])
        fit.maps[i].tmapunc = np.std(fit.maps[i].tmappost, axis=0)
        fit.maps[i].fmapunc = np.std(fit.maps[i].fmappost, axis=0)

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
        fit.scorr[i] = fit.maps[i].bestp[cfg.twod.ncurves+1]

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
        for i in range(fit.intens.shape[1]):
            check = np.sum(fit.intens[:,i] *
                           fit.maps[j].bestp[:cfg.twod.ncurves]) + \
                           fit.maps[j].bestp[cfg.twod.ncurves] / np.pi
            if check <= 0.0:
                msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                print(msg.format(fit.vislat[i], fit.vislon[i], check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    for j in range(len(fit.wlmid)):
        fmap, tmap = eigen.mkmaps(planet, fit.eigeny,
                                  fit.maps[j].bestp, cfg.twod.ncurves,
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
        plots.emaps(planet, fit.eigeny, cfg.outdir, proj='ortho')
        plots.emaps(planet, fit.eigeny, cfg.outdir, proj='rect')
        plots.emaps(planet, fit.eigeny, cfg.outdir, proj='moll')
        plots.lightcurves(fit.t, fit.lcs, cfg.outdir)
        plots.eigencurves(fit.t, fit.ecurves, cfg.outdir,
                          ncurves=cfg.twod.ncurves)
        plots.ecurvepower(fit.evalues, cfg.outdir)
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
        params, pstep, pmin, pmax = model.get_par(fit)

        # Override if specified by the user
        if hasattr(cfg.threed, 'params'):
            params = cfg.threed.params
        if hasattr(cfg.threed, 'pmin'):
            pmin   = cfg.threed.pmin
        if hasattr(cfg.threed, 'pmax'):
            pmax   = cfg.threed.pmax
        if hasattr(cfg.threed, 'pstep'):
            pstep  = cfg.threed.pstep

        # 5 filters, isobaric, cf fit symmetric, 12x24
        #params = np.array([-1.0872e+00,
        #                   -1.3168e+00,
        #                   -1.3139e+00,
        #                   -1.3998e+00,
        #                   -1.3303e+00,
        #                   491.3])
        # 5 filters, sinusoidal, cf fit symmetric, 12x24
        #params = np.array([-1.1496e+00, -9.5174e-02,  2.1525e-01, 32.132,
        #                   -1.4421e+00, -2.6572e-02,  9.6562e-02, 10.090,
        #                   -1.3452e+00,  1.8070e-02, -1.5103e-02, 12.320,
        #                   -1.3250e+00, -1.4248e-01,  3.1745e-02, 16.581,
        #                   -1.1958e+00, -8.8353e-02, -1.1020e-01, 19.863,
        #                   363.8])
        #pstep[3::4] = 0.0
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
                         pmax=pmax, pnames=cfg.threed.pnames,
                         leastsq=None, plots=cfg.threed.plots)

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
    fit.fluxgrid, fit.besttgrid, fit.taugrid, fit.p, fit.modelwngrid, fit.pmaps = \
        model.specgrid(fit.specbestp, fit)
    
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
        star, planet, system = utils.initsystem(fit)
        map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
        
    
        

    

    
