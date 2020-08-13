#! /usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import starry
sys.path.append('lib')
import pca

def main():
    # Eventually read configuration here. For now, will assign values

    # General settings
    lmax = 2

    # Star
    u = (0.5, 0.25)
    ms    = 1.0 # Mass in solar masses
    rs    = 1.0 # Radius in solar radii
    prots = 1.0 # Rotational period in days

    # Planet
    mp    =  0    # Mass in solar masses
    rp    =  0.1  # Radius in solar radii
    porb  =  1.0  # Orbital period in days
    protp =  1.0  # Rotational period in days
    Omega = 30    # Long. of asc. node in deg
    ecc   =  0.3  # Eccentricity
    w     = 30    # Long of periastron in deg
    t0    =  0    # Time of transit in days
    flux  =  0.01 # Flux relative to star

    # Observation
    t = np.linspace(-0.25, 0.75, 3000)

    # Create star object
    star = starry.Primary(starry.Map(ydeg=0, udeg=len(u), amp=1),
                          m=ms, r=rs, prot=prots)

    # Set limb-darkening
    star.map[1] = u[0]
    star.map[2] = u[1]

    # Create harmonic maps of the planet
    harmonics = []
    for i, l in enumerate(range(lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            harm = starry.kepler.Secondary(starry.Map(ydeg = lmax),
                                           m=mp, r=rp, porb=porb,
                                           prot=protp, Omega=Omega,
                                           ecc=ecc, w=w, t0=t0)

            if l > 0:
                harm.map[l, m] = flux

            harmonics.append(harm)

    # Create systems for each harmonic map
    systems = []
    for i in range(len(harmonics)):
        systems.append(starry.System(star, harmonics[i]))

    # Evaluate light curves
    lcs = np.zeros((len(systems), len(t)))
    for i, system in enumerate(systems):
        lcs[i] = system.flux(t).eval()

    evectors, score, evalues = pca.pca(lcs.T)

    

    
