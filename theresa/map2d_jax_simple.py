#!/usr/bin/env python3
"""
Simplified map2d using jaxoplanet - only computes positions at observation times
"""

import sys
import os
import jax.numpy as jnp
from jaxoplanet.starry.light_curves import light_curve
import numpy as np

# Add lib directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import fitclass as fc
from lib import temp as utils_jax  # This has our jaxoplanet initsystem
from lib import pca

def map2d(cfile):
    """
    Simplified version of map2d that:
    1. Reads the configuration file
    2. Reads the data
    3. Initializes the jaxoplanet system
    4. Computes planet and star positions at observation times

    Arguments
    ---------
    cfile : str
        Path to configuration file

    Returns
    -------
    fit : Fit object
        Fit object with loaded data and computed positions
    system : SurfaceSystem
        Jaxoplanet system object
    """

    # Create the master fit object
    fit = fc.Fit()

    print("="*60)
    print("SIMPLIFIED MAP2D WITH JAXOPLANET")
    print("="*60)

    print("\n[1/5] Reading the configuration file...")
    fit.read_config(cfile)
    cfg = fit.cfg
    print(f"  ✓ Config loaded from: {cfile}")
    print(f"  ✓ Star: M={cfg.star.m} Msun, R={cfg.star.r} Rsun")
    print(f"  ✓ Planet: M={cfg.planet.m} Msun, R={cfg.planet.r} Rsun")
    print(f"  ✓ Planet orbit: P={cfg.planet.porb} days, inc={cfg.planet.inc} deg")

    print("\n[2/5] Reading the data...")
    fit.read_data()
    print(f"  ✓ Number of datasets: {len(fit.datasets)}")
    for i, d in enumerate(fit.datasets):
        print(f"  ✓ Dataset {i+1}: {d.name}")
        print(f"    - Number of visits: {len(d.visits)}")
        total_times = sum(len(v.t) for v in d.visits)
        print(f"    - Total observation times: {total_times}")

    print("\n[3/5] Reading filters...")
    fit.read_filters()
    print("  ✓ Filter mean wavelengths (μm):")
    for d in fit.datasets:
        for i, wl in enumerate(d.wlmid):
            print(f"    - Filter {i+1}: {wl:.3f} μm")

    print("\n[4/5] Initializing jaxoplanet system...")
    # Use lmax=1 for simplicity (can change later)
    star_surface, planet_surface, system = utils_jax.initsystem(fit, ydeg=1)
    print(f"  ✓ Star surface created:")
    print(f"    - Radius: {star_surface.radius} Rsun")
    print(f"    - Period: {star_surface.period} days")
    print(f"  ✓ Planet surface created:")
    print(f"    - Radius: {planet_surface.radius} Rsun")
    print(f"    - Period: {planet_surface.period} days")
    print(f"    - Ylm degree: {planet_surface.y.deg}")
    print(f"  ✓ System created with {len(system.bodies)} planet(s)")

    print("\n[5/5] Computing planet and star positions at observation times...")
    for i, d in enumerate(fit.datasets):
        print(f"\n  Dataset {i+1}: {d.name}")

        # Get planet body (first orbiting body)
        planet_body = system.bodies[0]

        # Compute relative position of planet
        print(f"    Computing positions for {len(d.t)} time points...")
        x, y, z = planet_body.relative_position(d.t)

        flux = light_curve(system)(d.t).T[0]


        print("Calculating minimum and maximum observed longitudes.")
        d.minvislon, d.maxvislon = utils_jax.vislon(system, d)
        print("Minimum Longitude: {:6.2f}".format(d.minvislon))
        print("Maximum Longitude: {:6.2f}".format(d.maxvislon))


        # Store positions (convert to regular arrays if needed)
        d.x = jnp.array(x)
        d.y = jnp.array(y)
        d.z = jnp.array(z)

        print(f"    ✓ X positions: min={jnp.min(d.x):.3f}, max={jnp.max(d.x):.3f}")
        print(f"    ✓ Y positions: min={jnp.min(d.y):.3f}, max={jnp.max(d.y):.3f}")
        print(f"    ✓ Z positions: min={jnp.min(d.z):.3f}, max={jnp.max(d.z):.3f}")

        # Print some sample positions
        print(f"    Sample positions (first 5 times):")
        for j in range(min(5, len(d.t))):
            print(f"      t={d.t[j]:.3f} days: ({d.x[j]:.3f}, {d.y[j]:.3f}, {d.z[j]:.3f})")

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  - Positions stored in fit.datasets[i].x, .y, .z")
    print("  - System object available for light curve calculations")
    print("  - Ready to compute visibility and light curves")



    print("\n" + "="*60)
    print("Calculating uniform-map planet and star fluxes...")
    for d in fit.datasets:
        # Compute light curve with uniform map (Y00 only)
        # The system is already initialized with Y00=1.0 uniform map
        flux_result = light_curve(system)(d.t)
        d.sflux = jnp.array(flux_result.T[0])      # Star flux
        d.pflux_y00 = jnp.array(flux_result.T[1])  # Planet flux (uniform map)
        print(f"  ✓ Dataset {d.name}: computed {len(d.t)} flux points")

    print("\n" + "="*60)
    print("Calculating latitude and longitude of planetary grid...")
    cfg = fit.cfg
    fit.dlat = 180. / cfg.twod.nlat
    fit.dlon = 360. / cfg.twod.nlon
    fit.lat, fit.lon = jnp.meshgrid(jnp.linspace(-90  + fit.dlat / 2.,
                                                   90  - fit.dlat / 2.,
                                                   cfg.twod.nlat, endpoint=True),
                                     jnp.linspace(-180 + fit.dlon / 2.,
                                                   180 - fit.dlon / 2.,
                                                   cfg.twod.nlon, endpoint=True),
                                     indexing='ij')
    fit.dlatgrid, fit.dlongrid = jnp.meshgrid(jnp.ones(cfg.twod.nlat) * fit.dlat,
                                               jnp.ones(cfg.twod.nlon) * fit.dlon,
                                               indexing='ij')
    print(f"  ✓ Grid created: {cfg.twod.nlat} × {cfg.twod.nlon} points")
    print(f"  ✓ Lat range: [{jnp.min(fit.lat):.1f}, {jnp.max(fit.lat):.1f}] degrees")
    print(f"  ✓ Lon range: [{jnp.min(fit.lon):.1f}, {jnp.max(fit.lon):.1f}] degrees")

    print("\n" + "="*60)
    print("Optimizing 2D maps.")
    for d in fit.datasets:
        d.maps = []
        for i in range(len(d.wlmid)):
            print("{:.2f} um".format(d.wlmid[i]))
            m = fc.Map()

            d.maps.append(m)

            m.wlmid     = d.wlmid[i]
            m.filtwl    = d.filtwl[i]
            m.filtwn    = d.filtwn[i]
            m.filttrans = d.filttrans[i]
            m.flux      = d.flux[i]
            m.ferr      = d.ferr[i]

            # Where to put wl-specific outputs
            m.subdir = '{}-filt{}'.format(d.name, i+1)
            if not os.path.isdir(os.path.join(cfg.twod.outdir, m.subdir)):
                os.mkdir(os.path.join(cfg.twod.outdir, m.subdir))

            minbic = jnp.inf

            for l in range(1, cfg.twod.lmax+1):
                for n in range(0, cfg.twod.ncurves+1):
                    # Skip cases where n is higher than the number of
                    # available eigencurves, which is (l+1)**2, minus
                    # the uniform (l=0) case, since that's included by
                    # default
                    if n > (l+1)**2 - 1:
                        continue

                    # Also let's only do the n=0 case once, since
                    # it's exactly the same fit for every lmax.
                    # Link the LN objects for looping simplicity later
                    if l > 1 and n==0:
                        setattr(m, 'l{}n{}'.format(l, n), m.l1n0)
                        continue

                    print("Fitting lmax={}, n={}".format(l,n))
                    setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                    ln = getattr(m, 'l{}n{}'.format(l, n))

                    ln.subdir = 'l{}n{}'.format(l,n)

                    ln.wlmid = d.wlmid[i]

                    ln.ncurves = n
                    ln.lmax    = l

                    # New planet object with updated lmax
                    star, planet, system_ln = utils_jax.initsystem(fit, ln.lmax)

                    print("Running PCA to determine eigencurves.")
                    ncomp = ln.ncurves
                    if ln.ncurves == 0:
                        ncomp = None

                    # Call mkcurves from our jaxoplanet version in temp.py
                    ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        utils_jax.mkcurves(system_ln, d.t, ln.lmax,
                                          d.pflux_y00, ncurves=ncomp,
                                          method=cfg.twod.pca,
                                          orbcheck=cfg.twod.orbcheck,
                                          sigorb=cfg.twod.sigorb)
                
                    print("Calculating intensities of visible grid cells of each eigenmap.")
                    ln.intens, ln.vislat, ln.vislon = \
                        utils_jax.intensities(fit, d, ln)

                    # Save ln.intens to file for comparison
                    intens_file = os.path.join(cfg.twod.outdir, m.subdir, f'intensities_l{l}n{n}.npy')
                    np.save(intens_file, np.array(ln.intens))
                    print(f"  ✓ Saved intensities to {intens_file}")
                    print(f"    Shape: {ln.intens.shape}")
                    if ln.intens.size > 0:
                        print(f"    Min: {np.min(ln.intens):.6f}, Max: {np.max(ln.intens):.6f}")
                    else:
                        print(f"    Array is empty")

                    # Also save eigeny for comparison
                    eigeny_file = os.path.join(cfg.twod.outdir, m.subdir, f'eigeny_l{l}n{n}.npy')
                    np.save(eigeny_file, np.array(ln.eigeny))
                    print(f"  ✓ Saved eigeny to {eigeny_file}")
    return fit, system


if __name__ == "__main__":
    # Check if config file provided
    if len(sys.argv) < 2:
        print("Usage: python map2d_jax_simple.py <config_file>")
        print("\nExample:")
        print("  python map2d_jax_simple.py ../wasp76-example.cfg")
        sys.exit(1)

    cfile = sys.argv[1]

    # Check if file exists
    if not os.path.exists(cfile):
        print(f"Error: Config file not found: {cfile}")
        sys.exit(1)

    # Run simplified map2d
    fit, system = map2d(cfile)

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nVariables available:")
    print("  fit    - Fit object with loaded data")
    print("  system - Jaxoplanet SurfaceSystem object")
    print("\nAccess data:")
    print("  fit.datasets[0].x  - X positions")
    print("  fit.datasets[0].y  - Y positions")
    print("  fit.datasets[0].z  - Z positions")
    print("  fit.datasets[0].t  - Observation times")
    print("\nAccess system:")
    print("  system.bodies[0]         - Planet body")
    print("  system.body_surfaces[0]  - Planet surface")
    print("  system.central           - Star")
    print("  system.central_surface   - Star surface")
