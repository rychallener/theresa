#!/usr/bin/env python3
"""
Simplified map2d using jaxoplanet - only computes positions at observation times
"""

import sys
import os

# Set matplotlib to non-interactive backend to prevent plot windows from opening
import matplotlib


import jax.numpy as jnp
from jaxoplanet.starry.light_curves import light_curve
import numpy as np
import mc3
import time
# import matplotlib.pyplot as plt

# Add lib directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import fitclass as fc
from lib import temp as utils_jax  # This has our jaxoplanet initsystem
from lib import pca
from lib.jaxoplanet_logger import set_log_file
#from lib import model_2d as model  # Use model_2d to avoid theano dependencies

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

    # Set up logging for jaxoplanet comparisons
    log_file = os.path.join(os.path.dirname(cfile), "jaxoplanet_outputs.log")
    set_log_file(log_file)
    print(f"Logging jaxoplanet outputs to: {log_file}")

    print("="*60)
    print("SIMPLIFIED MAP2D WITH JAXOPLANET")
    print("="*60)

    # Start overall timing
    t_overall_start = time.time()

    print("\n[1/5] Reading the configuration file...")
    t_start = time.time()
    fit.read_config(cfile)
    cfg = fit.cfg
    t_elapsed = time.time() - t_start
    print(f"  ✓ Config loaded from: {cfile} (took {t_elapsed:.3f}s)")
    print(f"  ✓ Star: M={cfg.star.m} Msun, R={cfg.star.r} Rsun")
    print(f"  ✓ Planet: M={cfg.planet.m} Msun, R={cfg.planet.r} Rsun")
    print(f"  ✓ Planet orbit: P={cfg.planet.porb} days, inc={cfg.planet.inc} deg")

    print("\n[2/5] Reading the data...")
    t_start = time.time()
    fit.read_data()
    t_elapsed = time.time() - t_start
    print(f"  ✓ Number of datasets: {len(fit.datasets)} (took {t_elapsed:.3f}s)")
    for i, d in enumerate(fit.datasets):
        print(f"  ✓ Dataset {i+1}: {d.name}")
        print(f"    - Number of visits: {len(d.visits)}")
        total_times = sum(len(v.t) for v in d.visits)
        print(f"    - Total observation times: {total_times}")

    print("\n[3/5] Reading filters...")
    t_start = time.time()
    fit.read_filters()
    t_elapsed = time.time() - t_start
    print(f"  ✓ Filter mean wavelengths (μm): (took {t_elapsed:.3f}s)")
    for d in fit.datasets:
        for i, wl in enumerate(d.wlmid):
            print(f"    - Filter {i+1}: {wl:.3f} μm")

    print("\n[4/5] Initializing jaxoplanet system...")
    t_start = time.time()
    # Use lmax=1 for simplicity (can change later)
    star_surface, planet_surface, system = utils_jax.initsystem(fit, ydeg=1)
    t_elapsed = time.time() - t_start
    print(f"  ✓ Star surface created: (took {t_elapsed:.3f}s)")
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

        # Compute positions using system.position() like original starry implementation
        print(f"    Computing positions for {len(d.t)} time points...")
        t_pos_start = time.time()
        x_all, y_all, z_all = system.position(d.t)
        t_pos_elapsed = time.time() - t_pos_start
        print(f"      Position computation took {t_pos_elapsed:.3f}s")

        # Extract planet positions (first and only orbiting body)
        # Shape is (n_bodies, n_times), so we take [0] for the planet
        x_planet = jnp.array(x_all[0])
        y_planet = jnp.array(y_all[0])
        z_planet = jnp.array(z_all[0])

        # Star is at origin in observer frame (like starry)
        x_star = jnp.zeros_like(x_planet)
        y_star = jnp.zeros_like(y_planet)
        z_star = jnp.zeros_like(z_planet)

        # Store as tuples like starry: (star_array, planet_array)
        d.x = (x_star, x_planet)
        d.y = (y_star, y_planet)
        d.z = (z_star, z_planet)

        t_lc_start = time.time()
        flux = light_curve(system)(d.t).T[0]
        t_lc_elapsed = time.time() - t_lc_start
        print(f"      Light curve computation took {t_lc_elapsed:.3f}s")


        print("    Calculating minimum and maximum observed longitudes...")
        t_vislon_start = time.time()
        d.minvislon, d.maxvislon = utils_jax.vislon(system, d)
        t_vislon_elapsed = time.time() - t_vislon_start
        print("      Minimum Longitude: {:6.2f} (took {:.3f}s)".format(d.minvislon, t_vislon_elapsed))
        print("      Maximum Longitude: {:6.2f}".format(d.maxvislon))

        # d.x, d.y, d.z are now tuples: (star_array, planet_array)
        print(f"    ✓ Star X positions: min={jnp.min(d.x[0]):.3f}, max={jnp.max(d.x[0]):.3f}")
        print(f"    ✓ Star Y positions: min={jnp.min(d.y[0]):.3f}, max={jnp.max(d.y[0]):.3f}")
        print(f"    ✓ Star Z positions: min={jnp.min(d.z[0]):.3f}, max={jnp.max(d.z[0]):.3f}")
        print(f"    ✓ Planet X positions: min={jnp.min(d.x[1]):.3f}, max={jnp.max(d.x[1]):.3f}")
        print(f"    ✓ Planet Y positions: min={jnp.min(d.y[1]):.3f}, max={jnp.max(d.y[1]):.3f}")
        print(f"    ✓ Planet Z positions: min={jnp.min(d.z[1]):.3f}, max={jnp.max(d.z[1]):.3f}")

        # Print some sample positions
        print(f"    Sample planet positions (first 5 times):")
        for j in range(min(5, len(d.t))):
            print(f"      t={d.t[j]:.3f} days: ({d.x[1][j]:.3f}, {d.y[1][j]:.3f}, {d.z[1][j]:.3f})")

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  - Positions stored in fit.datasets[i].x, .y, .z")
    print("  - System object available for light curve calculations")
    print("  - Ready to compute visibility and light curves")



    print("\n" + "="*60)
    print("Calculating uniform-map planet and star fluxes...")
    t_start = time.time()
    for d in fit.datasets:
        # Compute light curve with uniform map (Y00 only)
        # The system is already initialized with Y00=1.0 uniform map
        flux_result = light_curve(system)(d.t)
        d.sflux = np.array(flux_result.T[0])      # Star flux (convert to numpy for numba)
        d.pflux_y00 = np.array(flux_result.T[1])  # Planet flux (convert to numpy for numba)
        print(f"  ✓ Dataset {d.name}: computed {len(d.t)} flux points")
    t_elapsed = time.time() - t_start
    print(f"  Total time for uniform flux calculation: {t_elapsed:.3f}s")

    print("\n" + "="*60)
    print("Calculating latitude and longitude of planetary grid...")
    t_start = time.time()
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
    t_elapsed = time.time() - t_start
    print(f"  ✓ Grid created: {cfg.twod.nlat} × {cfg.twod.nlon} points (took {t_elapsed:.3f}s)")
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
                    t_ln_start = time.time()
                    setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                    ln = getattr(m, 'l{}n{}'.format(l, n))

                    ln.subdir = 'l{}n{}'.format(l,n)

                    ln.wlmid = d.wlmid[i]

                    ln.ncurves = n
                    ln.lmax    = l

                    # New planet object with updated lmax
                    print("    Initializing system with lmax={}...".format(l))
                    t_init_start = time.time()
                    star_surface, planet_surface, system_ln = utils_jax.initsystem(fit, ln.lmax)
                    t_init_elapsed = time.time() - t_init_start
                    print("      System initialization took {:.3f}s".format(t_init_elapsed))

                    print("    Running PCA to determine eigencurves...")
                    ncomp = ln.ncurves
                    if ln.ncurves == 0:
                        ncomp = None

                    # Call mkcurves from our jaxoplanet version in temp.py
                    t_mkcurves_start = time.time()
                    ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        utils_jax.mkcurves(system_ln, d.t, ln.lmax,
                                          d.pflux_y00, ncurves=ncomp,
                                          method=cfg.twod.pca,
                                          orbcheck=cfg.twod.orbcheck,
                                          sigorb=cfg.twod.sigorb)
                    t_mkcurves_elapsed = time.time() - t_mkcurves_start
                    print("      PCA/mkcurves took {:.3f}s".format(t_mkcurves_elapsed))

                    print("    Calculating intensities of visible grid cells of each eigenmap...")
                    t_intens_start = time.time()
                    ln.intens, ln.vislat, ln.vislon = \
                        utils_jax.intensities(fit, d, ln)
                    t_intens_elapsed = time.time() - t_intens_start
                    print("      Intensities calculation took {:.3f}s".format(t_intens_elapsed))

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

                    # DIAGNOSTIC MODE: Exit early to see timing without MCMC
                    if False:  # Set to False to run full pipeline including MCMC
                        t_ln_elapsed = time.time() - t_ln_start
                        print("  >> Time for lmax={}, n={}: {:.3f}s (up to intensities, MCMC skipped)".format(l, n, t_ln_elapsed))
                        print("\n" + "="*60)
                        print("DIAGNOSTIC MODE: Stopping before MCMC")
                        print("="*60)
                        t_overall_elapsed = time.time() - t_overall_start
                        print(f"Total execution time (up to intensities): {t_overall_elapsed:.3f}s ({t_overall_elapsed/60:.2f} minutes)")
                        continue

                    # Set up for MCMC

                    if cfg.twod.posflux:
                        intens = ln.intens
                    else:
                        intens = None

                    params, pstep, pmin, pmax, pnames, texnames, pindex = \
                        utils_jax.get_par_2d(fit, d, ln)

                    baselines = tuple(v.baseline for v in d.visits)

                    # Convert to numpy arrays for numba JIT compatibility
                    tlocs = tuple(np.array(v.tloc) for v in d.visits)
                    dvecs = tuple(np.array(v.dvec) for v in d.visits)

                    indparams = (np.array(ln.ecurves), np.array(d.t),
                                 d.pflux_y00, d.sflux,
                                 ln.ncurves, intens, pindex,
                                 baselines, tlocs, dvecs)

                    # Better initial guess if possible
                    if hasattr(m, "l{}n{}".format(l,n-1)):
                        params = getattr(m, "l{}n{}".format(l,n-1)).bestp
                        params = np.insert(params, n-1, 0.0)

                    mc3data = d.flux[i]
                    mc3unc  = d.ferr[i]
                    mc3npz = os.path.join(cfg.twod.outdir,
                                          m.subdir,
                                          ln.subdir,
                                          '2dmcmc-l{}n{}-{:.2f}um.npz'.format(
                                              l,
                                              n,
                                              d.wlmid[i]))


                    # Make sure we don't use too much RAM
                    thinning = int(np.max((10, cfg.twod.nsamples // 1e5)))
                    mc3out = mc3.sample(data=mc3data, uncert=mc3unc,
                                        func=utils_jax.fit_2d,
                                        nsamples=cfg.twod.nsamples,
                                        burnin=cfg.twod.burnin,
                                        ncpu=cfg.twod.ncpu, sampler='snooker',
                                        savefile=mc3npz, params=params,
                                        indparams=indparams, pstep=pstep,
                                        leastsq=cfg.twod.leastsq,
                                        plots=cfg.twod.plots, pmin=pmin,
                                        pmax=pmax, pnames=pnames,
                                        texnames=texnames, thinning=thinning,
                                        fgamma=cfg.twod.fgamma,
                                        grbreak=1.01)

                    # MC3 doesn't clear its plots >:(
                    # plt.close('all')

                    ln.bestfit = mc3out['best_model']
                    ln.bestp   = mc3out['bestp']
                    ln.stdp    = mc3out['stdp']
                    ln.chisq   = mc3out['best_chisq']
                    ln.post    = mc3out['posterior']
                    ln.zmask   = mc3out['zmask']

                    # Isolate systematics models (used later in 3d mapping)
                    # Do this by calculating the best-fitting model without
                    # systematics and dividing it out of the best-fitting
                    # model
                    nobaselines = tuple('none' for v in d.visits)
                    nodvecs     = \
                        tuple(np.zeros((len(v.t), 1),
                                       dtype=float).T for v in d.visits)

                    nosysmodel = utils_jax.fit_2d(ln.bestp, np.array(ln.ecurves),
                                                  np.array(d.t), d.pflux_y00,
                                                  d.sflux, ln.ncurves,
                                                  intens, pindex,
                                                  nobaselines, tlocs,
                                                  nodvecs)

                    ln.systematics = ln.bestfit / nosysmodel

                    niter, nfree = ln.post.shape
                    nparams = len(params)
                    for ip in range(nparams):
                        if pstep[ip] == 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * params[ip],
                                axis=1)
                        if pstep[ip] < 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * ln.bestp[-int(pstep[ip])],
                                axis=1)

                    ln.nfreep = np.sum(pstep > 0)
                    ln.ndata  = mc3data.size

                    ln.redchisq = ln.chisq / \
                        (ln.ndata - ln.nfreep)
                    ln.bic      = ln.chisq + \
                        ln.nfreep * np.log(ln.ndata)

                    print("Chisq:         {}".format(ln.chisq))
                    print("Reduced Chisq: {}".format(ln.redchisq))
                    print("BIC:           {}".format(ln.bic))

                    if ln.bic < minbic:
                        minbic = ln.bic
                        m.bestln = ln

                    t_ln_elapsed = time.time() - t_ln_start
                    print("  >> Total time for lmax={}, n={}: {:.3f}s (excluding MCMC)".format(l, n, t_ln_elapsed))

            # TODO: Port utils.hotspotloc_driver to jaxoplanet
            # print("Calculating hotspot latitude and longitude.")
            # hs = utils.hotspotloc_driver(fit, m.bestln)
            # m.hslocbest  = hs[0]
            # m.hslocstd   = hs[1]
            # m.hslocpost  = hs[2]
            # m.hsloctserr = hs[3]
            #
            # msg = "Hotspot Longitude: {:.2f} +{:.2f} {:.2f}"
            # print(msg.format(m.hslocbest[1],
            #                  m.hsloctserr[1][0],
            #                  m.hsloctserr[1][1]))

            # Populate blackbody spectra outside posterior map
            # calculation loop for speed
            # Note: numerical issues can occur below 50 K, but it's
            #       possible that the model returns a map with fluxes
            #       low enough for such cold temperatures, which
            #       could result in issues in the future.
            if fit.cfg.star.starspec == 'custom':
                # Temperatures for later interpolation
                m.trange = np.linspace(50, 5000, 10000)
                # Blackbody spectra at each temperature
                bbs = utils_jax.blackbody_wl(m.trange, m.filtwl * 1e-6)
                # Interpolated stellar spectrum
                sspec_int = np.interp(m.filtwl, fit.starwl, fit.starflux)
                # Band-integrated stellar spectrum
                sspec_fint = np.trapz(m.filttrans * sspec_int,
                                      m.filtwl * 1e-6)
                rprs2 = (fit.cfg.planet.r / fit.cfg.star.r)**2
                fpfs_for_bbs = rprs2 * bbs / sspec_int
                m.fpfs_for_interp = np.trapz(
                    fpfs_for_bbs * m.filttrans * sspec_int,
                    m.filtwl * 1e-6, axis=1) / sspec_fint
                
            else:
                m.trange          = None
                m.fpfs_for_interp = None

            # TODO: Port utils.tmappost to jaxoplanet
            # print("Calculating flux and temperature map uncertainties.")
            # m.fmappost, m.tmappost = utils.tmappost(fit, m, m.bestln)
            # m.tmapunc = np.std(m.tmappost, axis=0)
            # m.fmapunc = np.std(m.fmappost, axis=0)

    print("Optimum lmax and ncurves:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  {:.2f} um: lmax={}, ncurves={}".format(m.wlmid,
                                                            m.bestln.lmax,
                                                            m.bestln.ncurves))
        
    # Save stellar correction terms (we need them later)
    #fit.scorr = np.zeros(len(fit.maps))
    #for i in range(len(fit.maps)):
    #    fit.scorr[i] = fit.maps[i].bestln.bestp[fit.maps[i].bestln.ncurves+1]

    print("Checking for negative fluxes in visible cells:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  Wl: {:.2f} um".format(m.wlmid))
            for i in range(m.bestln.intens.shape[1]):
                check = np.sum(m.bestln.intens[:,i] *
                               m.bestln.bestp[:m.bestln.ncurves]) + \
                               m.bestln.bestp[ m.bestln.ncurves] / np.pi
                if check <= 0.0:
                    msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                    print(msg.format(m.dataset.vislat[i],
                                     m.dataset.vislon[i],
                                     check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    t_start = time.time()
    for d in fit.datasets:
        for m in d.maps:
            star_surface, planet_surface, system_ln = utils_jax.initsystem(fit, m.bestln.lmax)
            # These are used or not used in mkmaps depending on the type
            # of stellar spectrum set in the configuration.
            fwl    = m.filtwl
            ftrans = m.filttrans
            swl    = fit.starwl if hasattr(fit, 'starwl') else None
            sspec  = fit.starflux if hasattr(fit, 'starflux') else None

            # Convert lat/lon from degrees to radians for jaxoplanet
            lat_rad = jnp.deg2rad(fit.lat)
            lon_rad = jnp.deg2rad(fit.lon)

            fmap, tmap = utils_jax.mkmaps(planet_surface, m.bestln.eigeny,
                                      m.bestln.bestp,
                                      m.bestln.ncurves, m.wlmid,
                                      cfg.star.r, cfg.planet.r,
                                      cfg.star.t, lat_rad, lon_rad,
                                      starspec=cfg.star.starspec,
                                      fwl=fwl, ftrans=ftrans, swl=swl,
                                      sspec=sspec)
            m.fmap = fmap
            m.tmap = tmap
    t_elapsed = time.time() - t_start
    print(f"  Total time for map construction: {t_elapsed:.3f}s")

    print("Temperature ranges of maps:")
    for d in fit.datasets:
        for m in d.maps:
            print("  {:.2f} um:".format(m.wlmid))
            tmax = np.max(m.tmap[~np.isnan(m.tmap)])
            tmin = np.min(m.tmap[~np.isnan(m.tmap)])
            print("    Max: {:.2f} K".format(tmax))
            print("    Min: {:.2f} K".format(tmin))
            print("    Negative: {:f}".format(np.sum(np.isnan(m.tmap))))

    # Make a single array of tmaps for convenience
    fit.nmaps = np.sum([len(d.maps) for d in fit.datasets])
    fit.tmaps2d = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    fit.fmaps2d = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))

    imap = 0
    for d in fit.datasets:
        for m in d.maps:
            fit.tmaps2d[imap] = m.tmap
            fit.fmaps2d[imap] = m.fmap
            imap += 1

    # Save fit object before plotting in case of crashes
    fit.save(cfg.twod.outdir)

    if cfg.twod.plots:
        print("Making plots.")
        # Note: Some plotting functions require starry planet objects which
        # are not available in jaxoplanet. These are commented out for now.
        for d in fit.datasets:
            continue
            for m in d.maps:
                outdir = os.path.join(cfg.twod.outdir, m.subdir)
                # Make sure the planet has the right lmax
                # star_surface, planet_surface, system_ln = utils_jax.initsystem(fit, m.bestln.lmax)
                # TODO: Port these plotting functions to work with jaxoplanet
                # plots.emaps(planet_surface, m.bestln.eigeny, outdir, proj='ortho')
                # plots.emaps(planet_surface, m.bestln.eigeny, outdir, proj='rect')
                # plots.emaps(planet_surface, m.bestln.eigeny, outdir, proj='moll')
                plots.lightcurves(d.t, m.bestln.lcs, outdir)
                plots.eigencurves(d.t, m.bestln.ecurves, outdir,
                                  ncurves=m.bestln.ncurves)
                plots.ecurvepower(m.bestln.evalues, outdir)

        #Commenting out plot code currently 
        
        #plots.pltmaps(fit)
        # plots.tmap_unc(fit)  # TODO: Requires tmapunc from utils.tmappost
        #plots.bestfit(fit, outdir=cfg.twod.outdir)
        #plots.ecurveweights(fit)
        # plots.hshist(fit)  # TODO: Requires hotspot data from utils.hotspotloc_driver
        #plots.bics(fit, outdir=cfg.twod.outdir)

    # With the new grid and visibility calculation moved to the 3D
    # function, these no longer function
    if cfg.twod.animations:
        pass
        #plots.visanimation(fit, outdir=cfg.twod.outdir)
        #plots.fluxmapanimation(fit, outdir=cfg.twod.outdir)

    # Print overall timing
    t_overall_elapsed = time.time() - t_overall_start
    print("\n" + "="*60)
    print("OVERALL TIMING SUMMARY")
    print("="*60)
    print(f"Total execution time: {t_overall_elapsed:.3f}s ({t_overall_elapsed/60:.2f} minutes)")

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
