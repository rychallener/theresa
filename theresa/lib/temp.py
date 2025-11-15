import numpy as np
import jax.numpy as jnp
import jax
import sys
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.orbit import SurfaceSystem, SurfaceBody
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.light_curves import light_curve
import scipy.constants as sc
import scipy.interpolate as spi
from numba import jit

from lib import pca
from lib import dummyfit
from lib.jaxoplanet_logger import get_logger
import matplotlib.pyplot as plt
from jaxoplanet.starry.visualization import show_surface

def initsystem(fit, ydeg):
    '''
    Uses a fit object to build the respective starry objects. Useful
    because starry objects cannot be pickled. Returns a tuple of
    (star, planet, system).
    '''
    
    cfg = fit.cfg
    star_ylm = Ylm.from_dense(jnp.array([1.0]), normalize=False)

    star_surface = Surface(
          y=star_ylm,
          inc=jnp.pi/2,              # Edge-on inclination
          period=cfg.star.prot,       # Rotation period in days
          radius=cfg.star.r,          # Radius in solar radii
          u=(),                       # No limb darkening
          normalize=False,            # No normalization
          amplitude=1.0               # Explicit amplitude
      )

      # Create planet surface with spherical harmonics up to ydeg
      # Initialize all coefficients to zero except Y_00 = 1.0 (uniform map)
    n_coeffs = (ydeg + 1)**2
    planet_ylm_coeffs = jnp.zeros(n_coeffs)
    planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)  # Y_00 = 1.0
    planet_ylm = Ylm.from_dense(planet_ylm_coeffs, normalize=False)

    planet_surface = Surface(
        y=planet_ylm,
        inc=jnp.deg2rad(cfg.planet.inc),     # Inclination in radians
        period=cfg.planet.prot,               # Rotation period in days
        radius=cfg.planet.r,                  # Radius in solar radii
        u=(),                                 # No limb darkening
        normalize=False,                      # No normalization
        amplitude=1.0,                        # Explicit amplitude
        phase=jnp.deg2rad(180)                # Initial rotation phase (theta0)
      )

      # Create the central star object
    central = Central(
        mass=cfg.star.m,      # Solar masses
        radius=cfg.star.r     # Solar radii
    )

    # Create the system with star as central body
    system = SurfaceSystem(
        central=central,
        central_surface=star_surface
    )

      # Add planet to the system
    system = system.add_body(
        period=cfg.planet.porb,               # Orbital period in days
        radius=cfg.planet.r,                  # Planet radius in solar radii
        mass=cfg.planet.m,                    # Planet mass in solar masses
        inclination=jnp.deg2rad(cfg.planet.inc),  # Orbital inclination
        eccentricity=cfg.planet.ecc,          # Eccentricity
        omega_peri=jnp.deg2rad(cfg.planet.w), # Argument of periastron
        asc_node=jnp.deg2rad(cfg.planet.Omega), # Longitude of ascending node
        time_transit=cfg.planet.t0,           # Time of transit
        surface=planet_surface                # Attach the planet surface
      )

    # Log the system initialization
    logger = get_logger()
    logger.log_function_call(
        "initsystem",
        inputs={
            "ydeg": ydeg,
            "star.m": cfg.star.m,
            "star.r": cfg.star.r,
            "star.prot": cfg.star.prot,
            "planet.m": cfg.planet.m,
            "planet.r": cfg.planet.r,
            "planet.inc": cfg.planet.inc,
            "planet.porb": cfg.planet.porb,
            "planet.prot": cfg.planet.prot,
            "planet.ecc": cfg.planet.ecc,
            "planet.w": cfg.planet.w,
            "planet.Omega": cfg.planet.Omega,
            "planet.t0": cfg.planet.t0
        },
        outputs={
            "star_surface.radius": star_surface.radius,
            "star_surface.period": star_surface.period,
            "planet_surface.radius": planet_surface.radius,
            "planet_surface.period": planet_surface.period,
            "planet_surface.inc": planet_surface.inc,
            "planet_surface.phase": planet_surface.phase,
            "planet_ylm_coeffs": planet_ylm_coeffs,
            "system.n_bodies": len(system.bodies)
        }
    )

    return star_surface, planet_surface, system


def vislon(system, data):
      import time
      start_time = time.time()

      t = data.t

      # Extract from system
      planet_surface = system.body_surfaces[0]
      planet_body = system.bodies[0]

      porb = planet_body.period
      prot = planet_surface.period
      t0 = planet_body.time_transit
      theta0 = jnp.rad2deg(planet_surface.phase)

      centlon = theta0 - (t - t0) / prot * 360
      limb1 = centlon - 90
      limb2 = centlon + 90
      limb1 = (limb1 + 180) % 360 - 180
      limb2 = (limb2 + 180) % 360 - 180

      elapsed = time.time() - start_time
      print(f"    vislon computation took {elapsed:.3f}s")

      minvislon = float(jnp.min(limb1))
      maxvislon = float(jnp.max(limb2))

      # Log visible longitude computation
      logger = get_logger()
      logger.log_function_call(
          "vislon",
          inputs={
              "t_shape": t.shape,
              "t_min": float(jnp.min(t)),
              "t_max": float(jnp.max(t)),
              "porb": porb,
              "prot": prot,
              "t0": t0,
              "theta0": theta0
          },
          outputs={
              "minvislon": minvislon,
              "maxvislon": maxvislon,
              "centlon_first": float(centlon[0]),
              "centlon_last": float(centlon[-1])
          }
      )

      return minvislon, maxvislon

def mkcurves(system, t, lmax, y00, ncurves=None, method='pca',
             orbcheck=None, sigorb=None):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax.

    Arguments
    ---------
    system: object
        A starry system object, initialized with a star and a planet

    t: 1D array
        Array of times at which to calculate eigencurves

    lmax: integer
        Maximum l to use in spherical harmonic maps

    y00: 1D array
        Light curve of a normalized, uniform map

    Returns
    -------
    eigeny: 2D array
        nharm x ny array of y coefficients for each harmonic. nharm is
        the number of harmonics, including positive and negative versions
        and excluding Y00. That is, 2 * ((lmax + 1)**2 - 1). ny is the
        number of y coefficients to describe a harmonic with degree lmax.
        That is, (lmax + 1)**2.

    evalues: 1D array
        nharm length array of eigenvalues

    evectors: 2D array
        nharm x nt array of normalized (unit) eigenvectors

    proj: 2D array
        nharm x nt array of the data projected in the new space (the PCA
        "eigencurves"). The imaginary part is discarded, if nonzero.
    """
    # Get planet surface from system
    planet_surface = system.body_surfaces[0]
    planet_body = system.bodies[0]
    central = system.central
    central_surface = system.central_surface

    nt = len(t)

    # Jaxoplanet function to evaluate flux with modified Ylm coefficients
    def evalflux(yval, track_time=False):
        """
        Compute light curve for given Ylm coefficients.
        yval is array of coefficients (excluding Y00).
        Returns star flux and planet flux separately.
        """
        import time
        if track_time:
            sys_start = time.time()

        # Create full Ylm coefficient array (including Y00 = 1)
        n_coeffs = (lmax + 1)**2
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)  # Y00 = 1 (uniform)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)  # Higher order terms
        # Create new Ylm with these coefficients (no normalization to match STARRY)
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
        # Create new planet surface with updated Ylm
        new_planet_surface = Surface(
            y=new_ylm,
            inc=planet_surface.inc,
            period=planet_surface.period,
            radius=planet_surface.radius,
            u=(),
            normalize=False,
            amplitude=1.0,
            phase=planet_surface.phase
        )

        # Create new system directly (like initsystem)
        new_system = SurfaceSystem(
            central=central,
            central_surface=central_surface
        )

        # Add planet to the system with the new surface
        new_system = new_system.add_body(
            period=planet_body.period,
            radius=planet_body.radius,
            mass=planet_body.mass,
            inclination=planet_body.inclination,
            eccentricity=planet_body.eccentricity,
            omega_peri=planet_body.omega_peri,
            time_transit=planet_body.time_transit,
            surface=new_planet_surface
        )
        print(planet_body.eccentricity)
        if track_time:
            sys_time = time.time() - sys_start
            lc_start = time.time()

        # Compute light curve - returns array with shape (n_bodies, n_times)
        print(t)


        flux_result = light_curve(new_system, order=20)(t)
        if track_time:
            lc_time = time.time() - lc_start
            print(f"    [evalflux breakdown] System creation: {sys_time:.3f}s, Light curve eval: {lc_time:.3f}s")

        # Extract star and planet fluxes
        # flux_result.T[0] is star, flux_result.T[1] is planet
        starflux = np.array(flux_result.T[0])
        planetflux = np.array(flux_result.T[1])
        print(ylm_coeffs)
        print(planetflux)
        print("------")
        print(starflux)
        plt.plot(t, planetflux)
        plt.plot(t, y00)
        plt.plot(t, planetflux -y00)


   
            #circle = plt.Circle((x, y), radius_ratio, color="k", fill=True, zorder=10)
        fig,ax = plt.subplots()
        show_surface(new_planet_surface, ax=ax, theta=0)

        plt.show()


        return starflux, planetflux

    # Create harmonic maps of the planet, excluding Y00
    # (lmax**2 maps, plus a negative version for all but Y00)
    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ilc = 0

    import time
    print(f"Computing {nharm} light curves for lmax={lmax} at {nt} time points...")
    start_time = time.time()

    # Track time spent in different parts
    system_creation_time = 0.0
    light_curve_eval_time = 0.0

    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            # Create array of Ylm coefficients (excluding Y00)
            yval = np.zeros(nharm // 2)

            # Set this specific harmonic to +1.0
            yval[ilc // 2] = 1.0
            lc_start = time.time()
            # Track detailed timing for the first light curve
            track = (ilc == 0)
            sflux, lcs[ilc] = evalflux(yval, track_time=track)
            lc_time = time.time() - lc_start

            # Set this specific harmonic to -1.0
            yval[ilc // 2] = -1.0
            sflux, lcs[ilc+1] = evalflux(yval, track_time=False)
            ilc += 2

            if (ilc // 2) % 5 == 0 or ilc == 2:
                print(f"  Computed light curves {ilc-1}/{nharm} (last pair took {lc_time:.3f}s)")

    total_time = time.time() - start_time
    print(f"Total light curve computation time: {total_time:.2f}s ({total_time/nharm:.3f}s per curve)")

    # If user wants to include additional eigencurves which explore
    # different orbital parameters
    if orbcheck is not None:
        # TODO: Implement orbcheck for jaxoplanet
        # For now, skip this feature
        print("Warning: orbcheck not yet implemented for jaxoplanet, skipping...")

    # Subtract uniform map contribution (jaxoplanet includes this in all light curves)
    print("Before: " + str(lcs))
    lcs -= y00
    
    # Additional correction: remove any remaining DC offset from each light curve
    # JAXOPLANET and STARRY compute harmonic contributions slightly differently.
    # This ensures each harmonic is orthogonal to Y00, matching STARRY's behavior.
    for i in range(lcs.shape[0]):
        lcs[i] -= np.mean(lcs[i])


    print("After: " + str(lcs))


    # Run PCA to determine orthogonal light curves
    if ncurves is None:
        ncurves = nharm
        if method == 'tsvd':
            ncurves -= 1

    print(f"Running PCA with method={method}, ncurves={ncurves}...")
    pca_start = time.time()
    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)
    pca_time = time.time() - pca_start
    print(f"  PCA computation took {pca_time:.3f}s")

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)

    # Convert orthogonal light curves into maps
    print(f"Converting eigenvectors to eigenmap coefficients...")
    conv_start = time.time()
    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps
    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                # (ok because evectors has only been sorted along
                #  one dimension)
                eigeny[j,yi] = evectors.T[j,shi] - evectors.T[j,shi+1]
                yi  += 1
                shi += 2
    conv_time = time.time() - conv_start
    print(f"  Eigenmap conversion took {conv_time:.3f}s")


    #for i in range (proj.shape[0]):
     #   plt.plot(proj[i])
      #  print(proj[i])
       # plt.xlabel("Jax")
       # plt.show()

    # Log mkcurves outputs
    logger = get_logger()
    logger.log_function_call(
        "mkcurves",
        inputs={
            "t_shape": t.shape,
            "lmax": lmax,
            "y00_shape": y00.shape,
            "ncurves": ncurves,
            "method": method
        },
        outputs={
            "eigeny": eigeny,
            "evalues": evalues,
            "evectors_shape": evectors.shape,
            "proj": proj,
            "lcs": lcs
        },
        notes=f"Computed {nharm} light curves, performed PCA to get {ncurves} eigencurves"
    )

    return eigeny, evalues, evectors, proj, lcs


def intensities(fit, data, ln):
    # We reinitialize the planet object here because the yval
    # assignments in the mkcurves theano function are tracked, so if
    # we don't pass that yval into the theano function here (and why
    # would we), theano gets confused as it runs through those
    # assignments in the graph. Perhaps there's a more elegant
    # solution.
    import time

    print(f"  Reinitializing system for intensities computation...")
    init_start = time.time()
    star_surface, planet_surface, system = initsystem(fit, ln.lmax)
    init_time = time.time() - init_start
    print(f"    System reinitialization took {init_time:.3f}s")

    grid_start = time.time()
    wherevis = np.where((np.array(fit.lon) + fit.dlon >= data.minvislon) &
                        (np.array(fit.lon) - fit.dlon <= data.maxvislon))

    vislon = jnp.deg2rad(np.array(fit.lon[wherevis].flatten()))
    vislat = jnp.deg2rad(np.array(fit.lat[wherevis].flatten()))

    nloc = len(vislon)
    grid_time = time.time() - grid_start
    print(f"    Visible grid calculation took {grid_time:.3f}s")

    intens = np.zeros((ln.ncurves, nloc))

    def evalintensity(yval):
        # Create full Ylm coefficient array (including Y00 = 1)
        n_coeffs = (ln.lmax + 1)**2
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)  # Y00 = 1 (uniform)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)  # Higher order terms

        # Use normalize=False to match how eigenmaps were created in mkcurves
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)
        new_planet_surface = Surface(
                    y=new_ylm,
                    inc=planet_surface.inc,
                    period=planet_surface.period,
                    radius=planet_surface.radius,
                    u=(),
                    normalize=False,
                    amplitude=1.0,
                    phase=planet_surface.phase
                )
        intensity  = new_planet_surface.intensity(vislat, vislon)
        uniform_ylm = Ylm.from_dense(jnp.array([1.0]), normalize=False)
        uniform_surface = Surface(
              y=uniform_ylm,
              inc=planet_surface.inc,
              period=planet_surface.period,
              radius=planet_surface.radius,
              u=(),
              normalize=False,
              amplitude=1.0,
              phase=planet_surface.phase
          )
        intensity -= uniform_surface.intensity(vislat, vislon)

        return intensity


    # JIT compile for speed (replaces Theano compilation)
    print(f"  Computing intensities for {ln.ncurves} eigenmaps at {nloc} visible locations...")
    start_time = time.time()

    jit_start = time.time()
    evalintensity_jit = jax.jit(evalintensity)
    jit_setup_time = time.time() - jit_start
    print(f"    JIT setup took {jit_setup_time:.3f}s")

    # Compute intensity for each eigenmap
    compile_time = 0.0
    for k in range(ln.ncurves):
        intens_start = time.time()
        intens[k] = np.array(evalintensity_jit(jnp.array(ln.eigeny[k, 1:])))
        intens_time = time.time() - intens_start
        if k == 0:
            compile_time = intens_time
            print(f"    First eigenmap (with JIT compilation): {compile_time:.3f}s")
        elif k == 1:
            print(f"    Second eigenmap (JIT compiled): {intens_time:.4f}s")

    total_time = time.time() - start_time
    avg_time = (total_time - compile_time) / max(1, ln.ncurves - 1) if ln.ncurves > 1 else 0
    print(f"  Total intensity computation time: {total_time:.2f}s (avg {avg_time:.4f}s per eigenmap after JIT)")

    # Log intensities computation
    logger = get_logger()
    logger.log_function_call(
        "intensities",
        inputs={
            "lmax": ln.lmax,
            "ncurves": ln.ncurves,
            "minvislon": data.minvislon,
            "maxvislon": data.maxvislon,
            "nloc": nloc,
            "eigeny_shape": ln.eigeny.shape
        },
        outputs={
            "intens": intens,
            "vislat": vislat,
            "vislon": vislon
        },
        notes=f"Computed intensities for {ln.ncurves} eigenmaps at {nloc} visible locations"
    )

    return intens, vislat, vislon

def mkmaps(planet_surface, eigeny, params, ncurves, wl, rs, rp, ts, lat, lon,
           starspec='bb', fwl=None, ftrans=None, swl=None, sspec=None):
    """
    Calculate flux map and brightness temperature map from
    a single 2D map fit.

    Arguments
    ---------
    planet_surface: Surface object
        Planet surface object from jaxoplanet. Will not be modified.

    eigeny: 2D array
        Eigenvalues for the eigenmaps that form the basis for the
        2D fit.

    params: 1D array
        Best-fitting parameters.

    ncurves: int
        Number of eigencurves (or eigenmaps) included in the total map.

    wl: 1D array
        The wavelength of the 2D map, in microns.

    rs: float
        Radius of the star (same units as rp)

    rp: float
        radius of the planet (same units as rs)

    ts: float
        Temperature of the star in Kelvin

    lat: 2d array
        Latitudes of grid to calculate map (in radians)

    lon: 2d array
        Longitudes of grid to calculate map (in radians)

    Returns
    -------
    fmap: 1D/2D array
        Array with shape matching lat and lon of planetary emission at
        each wavelength and location

    tmap: 1D/2D array
        Same as fmap but for brightness temperature.
    """
    import time
    start_time = time.time()

    fmap = np.zeros(lat.shape) # flux maps
    tmap = np.zeros(lat.shape) # temp maps

    # Infer lmax from eigeny shape
    lmax = int(np.sqrt(eigeny.shape[1])) - 1
    n_coeffs = (lmax + 1)**2

    print(f"Creating flux and temperature maps (grid: {lat.shape}, lmax={lmax}, ncurves={ncurves})...")

    # Uniform map term (Y00 only, scaled by params[ncurves])
    uniform_start = time.time()
    uniform_ylm = Ylm.from_dense(jnp.array([1.0]), normalize=False)
    uniform_surface = Surface(
        y=uniform_ylm,
        inc=planet_surface.inc,
        period=planet_surface.period,
        radius=planet_surface.radius,
        u=(),
        normalize=False,
        amplitude=1.0,
        phase=planet_surface.phase
    )
    fmap = np.array(uniform_surface.intensity(lat.flatten(), lon.flatten()).reshape(lat.shape)) * params[ncurves]
    uniform_time = time.time() - uniform_start

    # Combine scaled eigenmap Ylm terms
    combine_start = time.time()
    combined_ylm_coeffs = jnp.zeros(n_coeffs)
    combined_ylm_coeffs = combined_ylm_coeffs.at[0].set(1.0)  # Y00 = 1
    for i in range(ncurves):
        combined_ylm_coeffs = combined_ylm_coeffs.at[1:].add(eigeny[i, 1:] * params[i])

    # Create surface with combined coefficients
    combined_ylm = Ylm.from_dense(combined_ylm_coeffs, normalize=False)

    combined_surface = Surface(
        y=combined_ylm,
        inc=planet_surface.inc,
        period=planet_surface.period,
        radius=planet_surface.radius,
        u=(),
        normalize=False,
        amplitude=1.0,
        phase=planet_surface.phase
    )
    fmap += np.array(combined_surface.intensity(lat.flatten(), lon.flatten()).reshape(lat.shape))

    # Subtract extra Y00 map that jaxoplanet always includes
    fmap -= np.array(uniform_surface.intensity(lat.flatten(), lon.flatten()).reshape(lat.shape))
    combine_time = time.time() - combine_start

    # Convert to brightness temperatures
    # see Rauscher et al., 2018, Eq. 8
    tmap_start = time.time()
    tmap = fmap_to_tmap(fmap, wl, rp, rs, ts,
                              params[ncurves+1], starspec=starspec,
                              fwl=fwl, ftrans=ftrans, swl=swl,
                              sspec=sspec)
    tmap_time = time.time() - tmap_start

    total_time = time.time() - start_time
    flux_time = uniform_time + combine_time
    print(f"  Map creation time: {total_time:.3f}s")
    print(f"    - Uniform map: {uniform_time:.3f}s")
    print(f"    - Combined eigenmaps: {combine_time:.3f}s")
    print(f"    - Temperature conversion: {tmap_time:.3f}s")

    # Log map creation
    logger = get_logger()
    logger.log_function_call(
        "mkmaps",
        inputs={
            "eigeny_shape": eigeny.shape,
            "params_shape": params.shape,
            "ncurves": ncurves,
            "wl": wl,
            "lat_shape": lat.shape,
            "lon_shape": lon.shape,
            "starspec": starspec
        },
        outputs={
            "fmap": fmap,
            "tmap": tmap
        },
        notes=f"Created flux and temperature maps from {ncurves} eigenmaps"
    )

    return fmap, tmap

def fmap_to_tmap(fmap, meanwl, rp, rs, ts, scorr, starspec='bb',
                 fwl=None, ftrans=None, swl=None, sspec=None,
                 trange=None, fpfs_bb=None):
    '''
    Convert flux map to brightness temperatures.
    See Rauscher et al., 2018, eq. 8

    fmap: 2D array
        Array of star-normalized planet fluxes

    meanwl: Float
        Mean wavelength of planet fluxes, in microns.

    rp: Float
        Planet radius. Same units as rs.

    rs: Float
        Stellar radius. Same units as rp.

    ts: Float
        Stellar temperature (K)

    scorr: Float
        Stellar correction term. 

    starspec: String
        Three options:
            'bb' -- Blackbody evaluated at meanwl.
            'bbint' -- Blackbody, integrated over a filter.
            'custom' -- Provide stellar spectrum, which will be integrated.

    fwl: Array
        Array of filter wavelengths, in microns.

    ftrans: Array
        Array of filter transmission.

    swl: Array 
        Array of stellar spectrum wavelengths, in microns.

    sspec: Array
        Array of stellar spectrum, same units as the Planck function (mks)

    trange: 1D Array
        Array of temperatures corresponding to fpfs_bb

    fpfs_bb: 2D array
        Filter-integrated star-normalized planetary blackbody spectra at each 
        temperature in trange. Will be used to interpolate to temperatures
        using fmap. Calculated on the fly if not supplied. This can be
        very slow.
    '''
    meanwl_m = meanwl * 1e-6 # convert to m
    ptemp = (sc.h * sc.c) / (meanwl_m * sc.k)
    sfact = 1 + scorr
    if starspec == 'bb':
        tmap = ptemp / np.log(1 + (rp / rs)**2 *
                              (np.exp(ptemp / ts) - 1) /
                              (np.pi * fmap * sfact))
    elif starspec == 'bbint':
        if ((fwl is None) or
            (ftrans is None)):
            print('Must specify filter for integrated blackbody.')
        # Convert units
        fwl_m = fwl * 1e-6
        sbb = 2 * sc.h * sc.c**2 / fwl_m**5 / \
            (np.exp(sc.h * sc.c / fwl_m / sc.k / ts) -1 )
        sint = specint(fwl_m, sbb, [fwl_m], [ftrans])
        tmap = ptemp / np.log(1 + (rp / rs)**2 *
                              (2 * sc.h * sc.c**2 / meanwl_m**5) *
                              (1 / np.pi) *
                              (1 / (fmap * sfact)) *
                              (1 / sint))
    elif starspec == 'custom':
        if ((fwl is None) or
            (ftrans is None) or
            (sspec is None) or
            (swl is None)):
            print('Must specify filter and stellar spectrum.')

        if (trange is None) and (fpfs_bb is not None):
            print('Must specify temperatures if supplying fpfs_bb.')

        # Convert units
        fwl_m = fwl * 1e-6
        swl_m = swl * 1e-6
        
        if fpfs_bb is None:
            sspec_int = np.interp(fwl_m, swl_m, sspec)
            
            trange = np.linspace(50, 5000, 10000)
            bbs = blackbody_wl(trange, fwl_m)
            
            sspec_fint = np.trapz(ftrans * sspec_int, fwl_m)
            
            # Integrate over the filter throughput
            rprs2 = (rp / rs)**2
            fpfs_spec = rprs2 * bbs / sspec_int
            fpfs_bb = np.trapz(fpfs_spec * ftrans * sspec_int,
                               fwl_m, axis=1) / sspec_fint

        # Function to interpolate fluxes to temperatures
        interp_fpfs = spi.CubicSpline(fpfs_bb, trange)

        tmap = interp_fpfs(fmap * np.pi)
               
    return tmap



def specint(wn, spec, filtwn_list, filttrans_list):
    """
    Integrate a spectrum over the given filters.

    Arguments
    ---------
    wn: 1D array
        Wavenumbers (/cm) of the spectrum

    spec: 1D array
        Spectrum to be integrated

    filtwn_list: list
        List of arrays of filter wavenumbers, in /cm.

    filttrans_list: list
        List of arrays of filter transmission. Same length as filtwn_list.

    Returns
    -------
    intspec: 1D array
        The spectrum integrated over each filter. 
    """
    if len(filtwn_list) != len(filttrans_list):
        print("ERROR: list sizes do not match.")
        raise Exception
    
    intspec = np.zeros(len(filtwn_list)) 
    
    for i, (filtwn, filttrans) in enumerate(zip(filtwn_list, filttrans_list)):
        # Sort ascending
        idx = np.argsort(filtwn)
        
        intfunc = spi.interp1d(filtwn[idx], filttrans[idx],
                               bounds_error=False, fill_value=0)

        # Interpolate transmission
        inttrans = intfunc(wn)

        # Normalize to one
        norminttrans = inttrans / np.trapz(inttrans, wn)

        # Integrate filtered spectrum
        intspec[i] = np.trapz(spec * norminttrans, wn)

    return intspec

def blackbody_wl(T, wl):
    '''
    Calculates the Planck function for a grid of temperatures and
    wavelengths. Wavelenghts must be in m.
    '''
    bb = (2.0 * sc.h * sc.c**2 / (wl[np.newaxis]**5)) \
        * 1 / (np.exp(sc.h * sc.c / wl[np.newaxis] / sc.k / T[:, np.newaxis]) - 1.0)
    
    return bb



@jit(nopython=True)
def fit_2d(params, ecurves, t, y00, sflux, ncurves, intens, pindex,
           baselines, tlocs, dvecs):
    """
    Basic 2D fitting routine for a single wavelength.

    Arguments
    ---------
    params: 1D float array
        Model parameters, including the map parameters and
        ramp (baseline) parameters.

    ecurves: 2D float array
        Eigencurves that are used as the fitting basis for
        the planet map.

    t: 1D float array
        ALL the times associated with this planet map. If the
        map is being fit to multiple observations, this is
        a concatenated array of those times.

    y00: 1D float array
        The light curve contribution of the uniform map component.
        Same size as t.

    sflux: 1D float array
        The light curve contribution of the star (generally,
        1 everywhere). Same size as t.

    ncurves: Int
        The number of eigencurves to use in the fit.

    intens: 2D float array
        Precomputed eigenmap intensity, of size
        (ncurves x nlocs), where nlocs is the number of locations
        where the intensity has been precomputed. This array
        is used to determine if a fit has negative intensities
        on the map, and thus can be rejected. If intens is None,
        the model will not check for negative intensities.

    pindex: 2D boolean array
        Indices used to divide params between the models. E.g.,
        params[pindex[0]] pulls out the map parameters,
        params[pindex[1]] pulls out the ramp parameters for the
        first visit, etc.

    baselines: tuple of strings
        Ramp models to use for each visit.

    tlocs: list of 1D float arrays
        Local time (relative to start of visit) for each visit.
        Used for ramp model evaluation.

    dvecs: list of 2D float arrays
        Detrending vectors for each visit. This can be things
        like x-position, y-position, PSF-width, etc. Anything
        you think might be correlated with your light curve.
    """

    imodel = 0 # Keeps track of which model we are on
    mparams = params[pindex[imodel]]
    imodel += 1

    # Check for negative intensities
    if intens is not None:
        nloc = intens.shape[1]
        totint = np.zeros(nloc)
        for j in range(nloc):
            # Weighted eigenmap intensity
            totint[j] = np.sum(intens[:,j] * mparams[:ncurves])
            # Contribution from uniform map
            totint[j] += mparams[ncurves] / np.pi
        if np.any(totint <= 0):
            f = np.ones(len(t)) * np.min(totint)
            return f

    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * mparams[i]

    f += mparams[ncurves] * y00

    f += mparams[ncurves+1]

    # Renormalize (e.g., stellar variability between visits)
    istart = 0
    normparams = params[pindex[imodel]]
    imodel += 1
    for tloc, norm in zip(tlocs, normparams):
        f[istart:istart + len(tloc)] *= norm
        istart += len(tloc)

    f += sflux

    # Apply detrending vectors
    alldvec = np.zeros(len(t))
    istart = 0
    for dvec in dvecs:
        dmodel = np.ones(dvec.shape[1])
        for j, par in enumerate(params[pindex[imodel]]):
            dmodel += par * dvec[j]

        alldvec[istart:istart + dvec.shape[1]] += dmodel
        istart += dvec.shape[1]
        imodel += 1

    f *= alldvec

    # Apply ramps
    allramp = np.zeros(len(t))
    istart = 0
    for bl, tloc, ipar in zip(baselines, tlocs, pindex[imodel:]):
        rparams = params[ipar]
        if bl == 'none':
            ramp = np.ones(len(tloc))
        elif bl == 'linear':
            ramp = rparams[0] + rparams[1] * tloc
        elif bl == 'quadratic':
            ramp = rparams[0] +  rparams[1] * (tloc - rparams[3])**2 + \
                rparams[2] * (tloc - rparams[3])
        elif bl == 'sinusoidal':
            ramp = rparams[0] + rparams[1] * np.sin(
                2 * np.pi * tloc / rparams[2] - rparams[3])
        elif bl == 'exponential':
            ramp = rparams[0] + rparams[1] * np.exp((-rparams[2] * tloc) + rparams[3])
        elif bl == 'linexp':
            ramp = rparams[0] + rparams[1] * tloc + rparams[2] * \
                np.exp((1/rparams[3]) * -tloc)

        allramp[istart:istart + len(tloc)] += ramp
        istart += len(tloc)

    f *= allramp

    return f


def get_par_2d(fit, d, ln):
    '''
    Returns sensible parameter settings for each 2D model
    '''
    cfg = fit.cfg
    
    # Necessary parameters
    nmappar = ln.ncurves + 2

    params = np.zeros(nmappar)
    params[ln.ncurves] = 0.001
    
    pstep = np.ones(nmappar) *  0.01
    pmin  = np.ones(nmappar) * -1.0
    pmax  = np.ones(nmappar) *  1.0

    pstep[ln.ncurves+1] = 0.0

    pnames   = []
    texnames = []
    for j in range(ln.ncurves):
        pnames.append("C{}".format(j+1))
        texnames.append("$C_{{{}}}$".format(j+1))

    pnames.append("C0")
    texnames.append("$C_0$")

    pnames.append("scorr")
    texnames.append("$s_{corr}$")

    # Renormalize parameters
    nnormpar = len(d.visits)
    params   = np.concatenate((params,   np.repeat(1.0,  nnormpar)))
    pmin     = np.concatenate((pmin,     np.repeat(0.8,  nnormpar)))
    pmax     = np.concatenate((pmax,     np.repeat(1.2,  nnormpar)))
    pnames   = np.concatenate((pnames,   ['N{}'.format(i) for i in range(1, nnormpar+1)]))
    texnames = np.concatenate((texnames, ['$N_{}$'.format(i) for i in range(1, nnormpar+1)]))
    for v in d.visits:
        # Free parameter for renormalized visits,
        # fixed to 1.0 for non-remornalized visits.
        if v.renormalize:
            pstep = np.concatenate((pstep, (0.01,)))
        else:
            pstep = np.concatenate((pstep, (0.0,)))

    # Detrending vector coefficients
    ndvecpar = []
    for v in d.visits:
        if v.detrend:
            npar = v.dvec.shape[0]
            params   = np.concatenate((params,   np.repeat(0.0, npar)))
            pstep    = np.concatenate((pstep,    np.repeat(0.1, npar)))
            pmin     = np.concatenate((pmin,     np.repeat(-np.inf, npar)))
            pmax     = np.concatenate((pmax,     np.repeat( np.inf, npar)))
            pnames   = np.concatenate((pnames,   ['d{}'.format(i) for i in range(1, npar+1)]))
            texnames = np.concatenate((texnames, ['$d_{}$'.format(i) for i in range(1, npar+1)]))
        else:
            npar = 0

        ndvecpar.append(npar)
    
    nramppar = []

    # Parse baseline models
    for v in d.visits:
        if v.baseline == 'none':
            npar = 0
        elif v.baseline == 'linear':
            params   = np.concatenate((params,   (1.0, 0.0,)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001,)))
            pmin     = np.concatenate((pmin,     (0.8, -np.inf,)))
            pmax     = np.concatenate((pmax,     (1.2, np.inf,)))
            pnames   = np.concatenate((pnames,   ('b', 'm',)))
            texnames = np.concatenate((texnames, ('$b$', '$m$',)))
            npar = 2
        elif v.baseline == 'quadratic':
            params   = np.concatenate((params,   (1.0, 0.0,  0.0,   0.0)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01,  0.0)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0,  -1.0, -np.inf)))
            pmax     = np.concatenate((pmax,     (1.2, 1.0,   1.0,  np.inf)))
            pnames   = np.concatenate((pnames,   ('r0', 'r1',  'r2', 't0')))
            texnames = np.concatenate((texnames, ('r_0', '$r_1$', '$r_2$', '$t_0$')))
            npar = 3
        elif v.baseline == 'sinusoidal':
            params   = np.concatenate((params,   (1.0, -3.6e-5, 0.0885, 2.507)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001,    0.1)))
            pmin     = np.concatenate((pmin,     (0.8, -1.0,  0.05, -np.pi)))
            pmax     = np.concatenate((pmax,     (1.2, 1.0,  0.15,  np.pi)))
            pnames   = np.concatenate((pnames,   ('b', 'Amp.', 'Period', 'Phase')))
            texnames = np.concatenate((texnames, ('$b$', 'Amp.', 'Period', 'Phase')))
            npar = 4
        elif v.baseline == 'exponential':    
            params   = np.concatenate((params,   (1.0, 0.00001, 0.00001, 0.00001)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01,    0.01)))
            pmin     = np.concatenate((pmin,     (0.8, -5,  -5, -5)))
            pmax     = np.concatenate((pmax,     (1.2, 30, 30,  30))) 
            pnames   = np.concatenate((pnames,   ('r0', 'r1', 'r2', 'r3'))) 
            texnames = np.concatenate((texnames, ('$r_0$', '$r_1$', '$r_2$', '$r_3$')))
            npar = 4
        elif v.baseline == 'linexp':
            params   = np.concatenate((params,   (1.0, -0.00219881,0.00010304,0.01629347)))
            pstep    = np.concatenate((pstep,    (0.01, 0.001, 0.001, 0.001)))
            pmin     = np.concatenate((pmin,     (0.8, -1, -0.01, 0.0)))
            pmax     = np.concatenate((pmax,     (1.2, 1, 0.01, 0.2)))
            pnames   = np.concatenate((pnames,   ('b', 'm', 'A', 'tau')))
            texnames = np.concatenate((texnames, ('$b$', '$m$', '$A$', '$\\tau$')))
            npar = 4
        else:
            print("Unrecognized baseline model.")
            sys.exit()

        nramppar.append(npar)

    npar = np.concatenate(([nmappar], [nnormpar], ndvecpar, nramppar))
    totpar = np.sum(npar)
    cumpar = np.cumsum(npar)

    # Map model, normalization model, detrend model (per visit), ramp model (per visit)
    nmodel = 2 + 2 * len(d.visits)

    pindex = np.zeros((nmodel, totpar), dtype=bool)

    istart = 0
    for i in range(nmodel):
        where = np.where((np.arange(totpar) >= istart) &
                         (np.arange(totpar) <  cumpar[i]))
        pindex[i][where] = True
        istart += npar[i]

    return params, pstep, pmin, pmax, pnames, texnames, pindex

def main():
    fit = dummyfit.create_dummy_fit()

    star_surface, planet_surface, system = initsystem(fit, 3)
    #print(system.t)



    

main()