#! /usr/bin/env python3

# General imports
import os
import sys
import mc3
import pickle
import starry
import shutil
import subprocess
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
        eigen.mkcurves(system, fit.t, cfg.lmax)

    print("Calculating minimum and maximum observed longitudes.")
    fit.minvislon, fit.maxvislon = utils.vislon(planet, fit)
    print("Minimum Longitude: {:6.2f}".format(fit.minvislon))
    print("Maximum Longitude: {:6.2f}".format(fit.maxvislon))

    print("Calculating latitude and longitude of planetary grid.")
    # fit.lat, fit.lon = [a.eval() for a in \
    #                     planet.map.get_latlon_grid(res=cfg.res,
    #                                                projection='rect')] 

    # fit.dlat = fit.lat[1][0] - fit.lat[0][0]
    # fit.dlon = fit.lon[0][1] - fit.lon[0][0]

    fit.dlat = 180. / cfg.res
    fit.dlon = 360. / cfg.res
    fit.lat, fit.lon = np.meshgrid(np.linspace(-90  + fit.dlat / 2.,
                                                90  - fit.dlat / 2.,
                                               cfg.res, endpoint=True),
                                   np.linspace(-180 + fit.dlon / 2.,
                                                180 - fit.dlon / 2.,
                                               cfg.res, endpoint=True),
                                   indexing='ij')

    print("Calculating intensities of visible grid cells of each eigenmap.")
    fit.intens, fit.vislat, fit.vislon = eigen.intensities(planet, fit)
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    # Set up for MCMC
    if cfg.posflux:
        intens = fit.intens
    else:
        intens = None
        
    indparams = (fit.ecurves, fit.t, fit.wl, fit.pflux_y00,
                 fit.sflux, cfg.ncurves, intens)

    npar = cfg.ncurves + 2

    params = np.zeros(npar * len(fit.wl))
    params[cfg.ncurves::npar] = 0.001
    pstep  = np.ones( npar * len(fit.wl)) * 0.01

    mc3data = fit.flux.flatten()
    mc3unc  = fit.ferr.flatten()

    print("Optimizing 2D maps.")
    mc3out = mc3.fit(data=mc3data, uncert=mc3unc, func=model.fit_2d_wl,
                     params=params, indparams=indparams, pstep=pstep,
                     leastsq=cfg.leastsq)

    fit.bestfit = mc3out['best_model']
    fit.bestp   = mc3out['bestp']

    print("Best-fit parameters:")
    print(fit.bestp)

    # Save stellar correction terms (we need them later)
    # (there are ncurves+2 params per filter for each 2d fit,
    # and the stellar correction is the last term for each filter,
    # so we start from ncurves+2-1 (-1 for 0-start indexing)
    # and jump by ncurves+2)
    fit.scorr = fit.bestp[cfg.ncurves+1::cfg.ncurves+2]

    print("Calculating planet visibility with time.")
    nt, nlat, nlon = len(fit.t), len(fit.lat), len(fit.lon)
    fit.vis = np.zeros((nt, nlat, nlon))
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

    print("Checking critical locations:")
    for j in range(len(fit.wl)):
        print("  Wl: {} um".format(fit.wl[j]))
        for i in range(fit.intens.shape[1]):
            check = np.sum(fit.intens[:,i] *
                           fit.bestp[j*npar:j*npar+cfg.ncurves]) + \
                           fit.bestp[j*npar+cfg.ncurves] / np.pi
            msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
            print(msg.format(fit.vislat[i], fit.vislon[i], check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    fit.fmaps, fit.tmaps = eigen.mkmaps(planet, fit.eigeny, fit.bestp, npar,
                                        cfg.ncurves, fit.wl,
                                        cfg.star.r, cfg.planet.r, cfg.star.t,
                                        fit.lat, fit.lon)

    if cfg.plots:
        print("Making plots.")
        plots.circmaps(planet, fit.eigeny, cfg.outdir, ncurves=cfg.ncurves)
        plots.rectmaps(planet, fit.eigeny, cfg.outdir, ncurves=cfg.ncurves)
        plots.lightcurves(fit.t, fit.lcs, cfg.outdir)
        plots.eigencurves(fit.t, fit.ecurves, cfg.outdir, ncurves=cfg.ncurves)
        plots.ecurvepower(fit.evalues, cfg.outdir)
        plots.pltmaps(fit.tmaps, fit.wl, cfg.outdir, proj='rect')
        plots.bestfit(fit.t, fit.bestfit, fit.flux, fit.ferr, fit.wl,
                      cfg.outdir)

    if cfg.animations:
        print("Making animations.")
        plots.visanimation(fit)
        plots.fluxmapanimation(fit)

    fit.save(cfg.outdir)

def map3d(fit, system):
    cfg = fit.cfg
    print("Fitting spectrum.")
    if cfg.rtfunc == 'transit':
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

    elif cfg.rtfunc == 'taurex':
        fit.wngrid = np.arange(cfg.cfg.getfloat('taurex', 'wnlow'),
                               cfg.cfg.getfloat('taurex', 'wnhigh'),
                               cfg.cfg.getfloat('taurex', 'wndelt'))

        # Note: must do these things in the right order
        taurex.cache.OpacityCache().clear_cache()
        taurex.cache.OpacityCache().set_opacity_path(cfg.cfg.get('taurex',
                                                                 'csxdir'))
        taurex.cache.CIACache().set_cia_path(cfg.cfg.get('taurex',
                                                         'ciadir'))

        indparams = [fit, system]
        params = np.array([0.5, 0., -1., -1.5, -4., -2., -5., -5.5, -7.])
        pstep  = np.ones(len(params)) * 1e-3
        pmin   = np.ones(len(params)) * np.log10(cfg.ptop)
        pmax   = np.ones(len(params)) * np.log10(cfg.pbot)
        mc3npz = os.path.join(cfg.outdir, 'mcmc.npz')

        out = mc3.sample(data=fit.flux.flatten(), uncert=fit.ferr.flatten(),
                         func=model.sysflux, nsamples=cfg.nsamples,
                         burnin=cfg.burnin, ncpu=cfg.ncpu,
                         sampler='snooker', savefile=mc3npz,
                         params=params, indparams=indparams,
                         pstep=pstep, pmin=pmin, pmax=pmax,
                         leastsq=None, plots=cfg.plots)

    fit.specbestp = out['bestp']

    nfilt = len(cfg.filtfiles)
    nt    = len(fit.t)
    
    fit.specbestmodel = model.sysflux(fit.specbestp, fit, system)
    fit.specbestmodel = fit.specbestmodel.reshape((nfilt, nt))
        
    plots.bestfitlcsspec(fit)

    fit.besttgrid, fit.p = atm.tgrid(cfg.nlayers, cfg.res, fit.tmaps,
                                     10.**fit.specbestp, cfg.pbot,
                                     cfg.ptop, oob=cfg.oob)

    plots.bestfittgrid(fit)
    
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
        # Consider a more robust system (separate 2d and 3d sections
        # of config?)
        fit = fc.Fit()
        fit.read_config(cfile)
        fit = fc.load(outdir=fit.cfg.outdir)
        fit.read_config(cfile)
        #fit.read_data()
        star, planet, system = utils.initsystem(fit)
        map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
        
    
        

    

    
