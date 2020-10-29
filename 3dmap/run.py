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

def main(cfile):
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

    print("Calculating mean filter wavelengths.")
    fit.filtmean = utils.filtmean(fit.cfg.filtfiles)
    print(fit.filtmean)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star = starry.Primary(starry.Map(ydeg=1, amp=1),
                          m   =cfg.star.m,
                          r   =cfg.star.r,
                          prot=cfg.star.prot)

    planet = starry.kepler.Secondary(starry.Map(ydeg=cfg.lmax),
                                     m    =cfg.planet.m,
                                     r    =cfg.planet.r,
                                     porb =cfg.planet.porb,
                                     prot =cfg.planet.prot,
                                     Omega=cfg.planet.Omega,
                                     ecc  =cfg.planet.ecc,
                                     w    =cfg.planet.w,
                                     t0   =cfg.planet.t0,
                                     inc  =cfg.planet.inc,
                                     theta0=180)

    system = starry.System(star, planet)

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
    fit.lat, fit.lon = [a.eval() for a in \
                        planet.map.get_latlon_grid(res=cfg.res,
                                                   projection='rect')]

    print("Calculating intensities of visible grid cells of each eigenmap.")
    fit.intens, fit.vislat, fit.vislon = eigen.intensities(planet, fit)
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    if cfg.mkplots:
        print("Making plots.")
        plots.circmaps(planet, fit.eigeny, cfg.outdir, ncurves=cfg.ncurves)
        plots.rectmaps(planet, fit.eigeny, cfg.outdir, ncurves=cfg.ncurves)
        plots.lightcurves(fit.t, fit.lcs, cfg.outdir)
        plots.eigencurves(fit.t, fit.ecurves, cfg.outdir, ncurves=cfg.ncurves)
        plots.ecurvepower(fit.evalues, cfg.outdir)

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

    mc3npz = os.path.join(cfg.outdir, 'mcmc.npz')

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

    print("Calculating planet visibility with time.")
    nt, nlat, nlon = len(fit.t), len(fit.lat), len(fit.lon)
    fit.vis = np.zeros((nt, nlat, nlon))
    for it in range(len(fit.t)):
        print(it)
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

    print("Computing total flux and brightness temperature maps.")
    fit.fmaps, fit.tmaps = eigen.mkmaps(planet, fit.eigeny, fit.bestp, npar,
                                        cfg.ncurves, fit.wl,
                                        cfg.star.r, cfg.planet.r, cfg.star.t,
                                        res=cfg.res)
    
    fit.dlat = fit.lat[1][0] - fit.lat[0][0]
    fit.dlon = fit.lon[0][1] - fit.lon[0][0]

    if cfg.mkplots:
        plots.pltmaps(fit.tmaps, fit.wl, cfg.outdir, proj='rect')
        plots.bestfit(fit.t, fit.bestfit, fit.flux, fit.ferr, fit.wl,
                      cfg.outdir)

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
        params = np.array([1., 0., -1., -2., -3., -4., -5., -7., -8.])
        pstep  = np.ones(len(params)) * 1e-3
        pmin   = np.ones(len(params)) * np.log10(cfg.ptop)
        pmax   = np.ones(len(params)) * np.log10(cfg.pbot)

        out = mc3.sample(data=fit.flux.flatten(), uncert=fit.ferr.flatten(),
                         func=model.sysflux, nsamples=cfg.nsamples,
                         burnin=cfg.burnin, ncpu=cfg.ncpu,
                         sampler='snooker', savefile=mc3npz,
                         params=params, indparams=indparams,
                         pstep=pstep, pmin=pmin, pmax=pmax,
                         leastsq=None, plots=cfg.mkplots)

    fit.specbestp = out['bestp']
    fit.specbestmodel = model.fit_spec(fit.specbestp, fit)
        
    plots.bestfitspec(fit)

    fit.besttgrid, fit.p = atm.tgrid(cfg.nlayers, cfg.res, fit.tmaps,
                                     10.**fit.specbestp, cfg.pbot,
                                     cfg.ptop, oob=cfg.oob)

    plots.bestfittgrid(fit)
    
    fit.save(cfg.outdir)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide configuration file as a command-line argument.")
        sys.exit()
    else:
        cfile = sys.argv[1]
    main(cfile)
    
        

    

    
