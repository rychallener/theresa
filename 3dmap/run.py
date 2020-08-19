#! /usr/bin/env python3

import os
import sys
import mc3
import pickle
import starry
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('lib')
import pca
import eigen
import model
import plots
import config

starry.config.quiet = True

def main(cfile):
    """
    One function to rule them all.
    """
    # Create the master fit object
    fit = model.Fit()
    
    print("Reading the configuration file.")
    cfg = config.read_config(cfile)
    fit.cfg = cfg

    # Read data
    fit.t, fit.flux, fit.ferr = np.loadtxt(fit.cfg.datafile, unpack=True)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star = starry.Primary(starry.Map(ydeg=0, udeg=0, amp=1),
                          m   =cfg.cfg.getfloat('Star', 'm'),
                          r   =cfg.cfg.getfloat('Star', 'r'),
                          prot=cfg.cfg.getfloat('Star', 'prot'))

    planet = starry.kepler.Secondary(starry.Map(ydeg=cfg.lmax),
                                     m    =cfg.cfg.getfloat('Planet', 'm'),
                                     r    =cfg.cfg.getfloat('Planet', 'r'),
                                     porb =cfg.cfg.getfloat('Planet', 'porb'),
                                     prot =cfg.cfg.getfloat('Planet', 'prot'),
                                     Omega=cfg.cfg.getfloat('Planet', 'Omega'),
                                     ecc  =cfg.cfg.getfloat('Planet', 'ecc'),
                                     w    =cfg.cfg.getfloat('Planet', 'w'),
                                     t0   =cfg.cfg.getfloat('Planet', 't0'),
                                     inc  =cfg.cfg.getfloat('Planet', 'inc'))


    system = starry.System(star, planet)
    
    fit.sflux, fit.pflux_y00 = [a.eval() for a in  \
                                system.flux(fit.t, total=False)]

    print("Running PCA to determine eigencurves.")
    fit.eigeny, fit.evalues, fit.evectors, fit.ecurves, fit.lcs = \
        eigen.mkcurves(system, fit.t, cfg.lmax)
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    if cfg.mkplots:
        print("Making plots.")
        plots.circmaps(planet, fit.eigeny, cfg.outdir)
        plots.rectmaps(planet, fit.eigeny, cfg.outdir)
        plots.lightcurves(fit.t, fit.lcs, cfg.outdir)
        plots.eigencurves(fit.t, fit.ecurves, cfg.outdir, ncurves=cfg.ncurves)
        plots.ecurvepower(fit.evalues, cfg.outdir)

    indparams = (fit.ecurves, fit.t, fit.pflux_y00, fit.sflux, cfg.ncurves)

    params = np.zeros(cfg.ncurves + 2)
    pstep  = np.ones(cfg.ncurves + 2) * 0.01

    mc3npz = os.path.join(cfg.outdir, 'mcmc.npz')

    mc3out = mc3.sample(data=fit.flux, uncert=fit.ferr, func=model.fit_2d,
                        params=params, indparams=indparams, pstep=pstep,
                        sampler='snooker', nsamples=1000000, burnin=10000,
                        ncpu=4, savefile=mc3npz, plots=True)

    bestfit = mc3out['best_model']
    bestp   = mc3out['bestp']

    # TODO: build map out of best-fitting parameters by calculating
    # Y values and summing the eigencurve maps
    
    fit.save(fit.cfg.outdir)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide configuration file as a command-line argument.")
        sys.exit()
    else:
        cfile = sys.argv[1]
    main(cfile)
        

    

    
