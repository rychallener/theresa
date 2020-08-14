#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import starry
sys.path.append('lib')
import pca
import eigen

def main():
    # Eventually read configuration here. For now, will assign values

    # General settings
    lmax = 3
    outdir = 'testout'

    # Star
    ms    = 1.0 # Mass in solar masses
    rs    = 1.0 # Radius in solar radii
    prots = 1.0 # Rotational period in days

    # Planet
    mp    =  0.001 # Mass in solar masses
    rp    =  0.1   # Radius in solar radii
    porb  =  1.0   # Orbital period in days
    protp =  1.0   # Rotational period in days
    Omega =  0.0   # Long. of asc. node in deg
    ecc   =  0.0   # Eccentricity
    inc   = 88.5   # Inclination
    w     = 90     # Long of periastron in deg
    t0    =  0     # Time of transit in days

    # Observation
    nt = 1000
    t = np.linspace(0.4*porb, 0.6*porb, nt)

    # Create star, planet, and system objects
    star = starry.Primary(starry.Map(ydeg=0, udeg=0, amp=1),
                          m=ms, r=rs, prot=prots)

    planet = starry.kepler.Secondary(starry.Map(ydeg=lmax),
                                     m=mp, r=rp, porb=porb,
                                     prot=protp, Omega=Omega,
                                     ecc=ecc, w=w, t0=t0,
                                     inc=inc)

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

        fill = np.int(np.floor(np.log10(nharm))) + 1
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
    main()
        

    

    
