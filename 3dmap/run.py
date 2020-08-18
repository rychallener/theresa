#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import starry
sys.path.append('lib')
import pca
import eigen
import plots
import config

starry.config.quiet = True

def main(cfile):
    """
    One function to rule them all.
    """
    print("Reading the configuration file.")
    cfg = config.read_config(cfile)

    # Observation
    # TODO: Eventually read this in
    nt = 1000
    t = np.linspace(0.4, 0.6, nt)

    # Create star, planet, and system objects
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

    print("Running PCA to determine eigencurves.")
    eigeny, evalues, evectors, proj, lcs = eigen.mkcurves(system, t, cfg.lmax)
    
    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    if cfg.mkplots:
        print("Making plots.")
        plots.circmaps(planet, eigeny, cfg.outdir)
        plots.rectmaps(planet, eigeny, cfg.outdir)
        plots.lightcurves(t, lcs, cfg.outdir)
        plots.eigencurves(t, proj, cfg.outdir, ncurves=cfg.ncurves)
        plots.ecurvepower(evalues, cfg.outdir)

        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide configuration file as a command-line argument.")
        sys.exit()
    else:
        cfile = sys.argv[1]
    main(cfile)
        

    

    
