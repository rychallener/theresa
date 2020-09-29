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
    
    fit.sflux, fit.pflux_y00 = [a.eval() for a in  \
                                system.flux(fit.t, total=False)]

    print("Running PCA to determine eigencurves.")
    fit.eigeny, fit.evalues, fit.evectors, fit.ecurves, fit.lcs = \
        eigen.mkcurves(system, fit.t, cfg.lmax)

    print("Computing location of minimum and maximum of each eigenmap.")
    fit.mmlat, fit.mmlon, fit.mmint = eigen.emapminmax(planet, fit.eigeny,
                                                       cfg.ncurves)
    
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
        intens = fit.mmint
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
    
    # mc3out = mc3.sample(data=mc3data, uncert=mc3unc, func=model.fit_2d_wl,
    #                     params=params, indparams=indparams, pstep=pstep,
    #                     sampler='snooker', nsamples=cfg.nsamples,
    #                     burnin=cfg.burnin, ncpu=cfg.ncpu, savefile=mc3npz,
    #                     plots=True, leastsq=cfg.leastsq)

    fit.bestfit = mc3out['best_model']
    fit.bestp   = mc3out['bestp']

    print("Best-fit parameters:")
    print(fit.bestp)

    print("Computing total flux and brightness temperature maps.")
    fmaps, tmaps = eigen.mkmaps(planet, fit.eigeny, fit.bestp, npar,
                                cfg.ncurves, fit.wl,
                                cfg.star.r, cfg.planet.r, cfg.star.t,
                                res=cfg.res)
    
    if cfg.mkplots:
        plots.pltmaps(tmaps, fit.wl, cfg.outdir, proj='rect')
        plots.bestfit(fit.t, fit.bestfit, fit.flux, fit.ferr, fit.wl,
                      cfg.outdir)

    print("Initializing atmosphere.")
    pmaps = np.array([1e-3, 1e0])
    tgrid, p = atm.tgrid(cfg.nlayers, cfg.res, tmaps, pmaps, cfg.pbot,
                       cfg.ptop, kind='linear', bounds_error=False,
                       fill_value='extrapolate')
    
    r, p, abn, spec = atm.atminit(cfg.atmtype, cfg.atmfile,
                                  p, tgrid,
                                  cfg.planet.m, cfg.planet.r,
                                  cfg.planet.p0, cfg.elemfile,
                                  cfg.outdir)

    print("Generating spectrum.")
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

        for i in range(cfg.res):
            for j in range(cfg.res):
                rtt = TemperatureArray(
                    tp_array=tgrid[:,i,j])
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
                rtchem = taurex.chemistry.TaurexChemistry()
                for k in range(len(spec)):
                    if spec[k] not in ['H2', 'He']:
                        gas = trc.ArrayGas(spec[k], abn[k,:,i,j])
                        rtchem.addGas(gas)
                rtp = taurex.pressure.SimplePressureProfile(
                    nlayers=cfg.nlayers,
                    atm_min_pressure=cfg.ptop * 1e5,
                    atm_max_pressure=cfg.pbot * 1e5)
                rt = trc.EmissionModel3D(
                    planet=rtplan,
                    star=rtstar,
                    pressure_profile=rtp,
                    temperature_profile=rtt,
                    chemistry=rtchem,
                    nlayers=cfg.nlayers,
                    latmin=-np.pi/2.,
                    latmax=np.pi/2.,
                    lonmin=0,
                    lonmax=2*np.pi)
                rt.add_contribution(taurex.contributions.AbsorptionContribution())
                rt.add_contribution(taurex.contributions.CIAContribution())
                rt.build()
                if i == 0 and j == 0:
                    fit.flux = np.zeros((len(rt.nativeWavenumberGrid),
                                        cfg.res, cfg.res))
                fit.wn, fit.flux[:,i,j], tau, ex = rt.model()


    fit.save(fit.cfg.outdir)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide configuration file as a command-line argument.")
        sys.exit()
    else:
        cfile = sys.argv[1]
    main(cfile)
    
        

    

    
