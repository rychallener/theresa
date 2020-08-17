#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import starry
sys.path.append('lib')
import pca
import eigen
import config

def main(cfile):
    cfg = config.read_config(cfile)
    lmax   = cfg.cfg.getint('General', 'lmax')
    outdir = cfg.cfg.get(   'General', 'outdir')

    # Observation
    # TODO: Eventually read this in
    nt = 1000
    t = np.linspace(0.4, 0.6, nt)

    # Create star, planet, and system objects
    star = starry.Primary(starry.Map(ydeg=0, udeg=0, amp=1),
                          m   =cfg.cfg.getfloat('Star', 'm'),
                          r   =cfg.cfg.getfloat('Star', 'r'),
                          prot=cfg.cfg.getfloat('Star', 'prot'))

    planet = starry.kepler.Secondary(starry.Map(ydeg=lmax),
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

    eigeny, evalues, evectors, proj = eigen.mkcurves(system, t, lmax)
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    nharm = 2 * ((lmax + 1)**2 - 1)

    for j in range(nharm):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1

        fill = np.int(np.log10(nharm)) + 1
        fnum = str(j).zfill(fill)
        
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.imshow(planet.map.render(theta=180).eval(),
                  origin="lower", cmap="plasma")
        ax.axis("off")
        plt.savefig(os.path.join(outdir, 'emap-ecl-{}.png'.format(fnum)))
        plt.close(fig)

        fig, ax = plt.subplots(1, figsize=(6,3))
        ax.imshow(planet.map.render(projection="rect").eval(),
                  origin="lower", cmap="plasma")
        plt.savefig(os.path.join(outdir, 'emap-rect-{}.png'.format(fnum)))
        plt.close(fig)
                


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide configuration file as a command-line argument.")
        sys.exit()
    else:
        cfile = sys.argv[1]
    main(cfile)
        

    

    
