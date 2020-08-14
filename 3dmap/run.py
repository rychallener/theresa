#! /usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import starry
sys.path.append('lib')
import pca

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

    # Create harmonic maps of the planet, excluding Y00
    # (lmax**2 maps, plus a negative version for all but Y00)
    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ind = 0
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):           
            planet.map[l, m] =  1.0
            sflux, lcs[ind]   = [a.eval() for a in system.flux(t, total=False)]
            planet.map[l, m] = -1.0
            sflux, lcs[ind+1] = [a.eval() for a in system.flux(t, total=False)]
            planet.map[l, m] = 0.0
            ind += 2
            

    # Run PCA to determine orthogonal light curves
    evalues, evectors, proj = pca.pca(lcs)

    # Convert orthogonal light curves into a map
    eigeny = np.zeros((nharm, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps
    for j in range(nharm):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                eigeny[j,yi] = evectors.T[j,shi] - evectors.T[j,shi+1]
                yi  += 1
                shi += 2

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

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
        

    

    
