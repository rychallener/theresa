import numpy as np
import jax.numpy as jnp
import pickle
import time
import constants as c
import scipy.constants as sc
import scipy.interpolate as spi
import eigen
import progressbar
import mc3.stats as ms
from numba import njit
import matplotlib.pyplot as plt
import jax
from jaxoplanet_logger import get_logger

import jaxoplanet.starry.core                as core
import jaxoplanet.starry.orbit               as orbit
import jaxoplanet.starry.surface             as surface
import jaxoplanet.starry.ylm                 as ylm
import jaxoplanet.orbits.keplerian           as keplerian
import jaxoplanet.starry.light_curves        as light_curves
import jaxoplanet.starry.core.s2fft_rotation as s2fft_rotation
import jaxoplanet.starry.core.rotation       as rotation
import jaxoplanet.starry.surface_is_physical as surface_is_physical

def initsystem(fit, ydeg, y=None):
    '''
    Uses a fit object to build the respective jaxoplanet objects. Useful
    because objects cannot be pickled. Returns a tuple of
    (star, planet, system).
    '''
    
    cfg = fit.cfg
    star_ylm = ylm.Ylm.from_dense(jnp.array([1.0]), normalize=True)

    star_surface = surface.Surface(
          y=star_ylm,
          inc=jnp.pi/2,               # Edge-on inclination
          period=cfg.star.prot,       # Rotation period in days
          radius=cfg.star.r,          # Radius in solar radii
          amplitude=1.0               # Normalized amplitude
      )

    # Create planet surface with spherical harmonics up to ydeg
    # Initialize all coefficients to zero except Y_00 = 1.0 (uniform map)
    # unless given a specific array
    if y is None:
        n_coeffs = (ydeg + 1)**2
        planet_ylm_coeffs = jnp.zeros(n_coeffs)
        planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)  # Y_00 = 1.0
    else:
        planet_ylm_coeffs = jnp.array(y)
        planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)

    planet_ylm = ylm.Ylm.from_dense(planet_ylm_coeffs, normalize=True)

    planet_surface = surface.Surface(
        y=planet_ylm,
        inc=np.pi/2,            # Inclination in radians
        period=cfg.planet.prot, # Rotation period in days
        radius=cfg.planet.r,    # Radius in solar radii
        amplitude=1.0,
        phase=jnp.deg2rad(180)  # Initial rotation phase (theta0)
      )

    # Create the central star object
    central = keplerian.Central(
        mass=cfg.star.m,      # Solar masses
        radius=cfg.star.r     # Solar radii
    )

    # Create the system with star as central body
    system = orbit.SurfaceSystem(
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

    return star_surface, planet_surface, system


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

    
def vislon(system, data):
    """
    Determines the range of visible longitudes based on times of
    observation.

    Arguments
    ---------
    system: jaxoplanet System object
        System object

    data: Dataset object
        Dataset object. Must contain observation information.

    Returns
    -------
    minlon: float
        Minimum visible longitude, in degrees

    maxlon: float
        Maximum visible longitude, in degrees
    """
    t = data.t

    psurf = system.body_surfaces[0]
    pbody = system.bodies[0]

    porb   = pbody.period               # days / orbit
    prot   = psurf.period               # days / rotation
    t0     = pbody.time_transit         # days
    theta0 = psurf.phase * 180. / np.pi # degrees

    # Central longitude at each time ("sub-observer" point)
    centlon = theta0 - (t - t0) / prot * 360

    # Minimum and maximum longitudes (assuming +/- 90 degree
    # visibility)
    limb1 = centlon - 90
    limb2 = centlon + 90

    # Rescale to [-180, 180]
    limb1 = (limb1 + 180) % 360 - 180
    limb2 = (limb2 + 180) % 360 - 180

    return np.min(limb1), np.max(limb2)
  
    
def readfilters(filterfiles):
    """
    Reads filter files and determines the mean wavelength.
    
    Arguments
    ---------
    filterfiles: list
        list of paths to filter files

    Returns
    -------
    filtmid: 1D array
        Array of mean wavelengths
    """
    filtwl_list    = []
    filtwn_list    = []
    filttrans_list = []
    
    wnmid = np.zeros(len(filterfiles))
    for i, filterfile in enumerate(filterfiles):
        filtwl, trans = np.loadtxt(filterfile, unpack=True)
        
        filtwn = 1.0 / (filtwl * c.um2cm)

        wnmid[i] = np.sum(filtwn * trans) / np.sum(trans)

        filtwl_list.append(filtwl)
        filtwn_list.append(filtwn)
        filttrans_list.append(trans)

    wlmid = 1 / (c.um2cm * wnmid)

    return filtwl_list, filtwn_list, filttrans_list, wnmid, wlmid

def visibility(fit, d, lmax, sampling='Mollweide'):
    """
    Calculate the visibility of a grid of cells on a planet for
    a series of times using the design matrix and pixel transform.
    """
    t, x, y, z = d.t, d.x, d.y, d.z

    Nt = len(t)

    # Determine spatial sampling
    # Might be better to have user supply lat/lon (e.g., through
    # calling a different function) rather than compute them here.
    # Mollweide adapted from starry
    if sampling == 'Mollweide':
        npix = fit.cfg.threed.oversample * (lmax + 1)**2

        Ny = int(np.sqrt(npix * np.pi / 4.0))
        Nx = 2 * Ny

        y, x = np.meshgrid(
            np.sqrt(2) * np.linspace(-1, 1, Ny),
            2 * np.sqrt(2) * np.linspace(-1, 1, Nx),
        )
        x = x.flatten()
        y = y.flatten()

        # Remove off-grid points
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        idx = (y / a) ** 2 + (x / b) ** 2 <= 1
        y = y[idx]
        x = x[idx]

        # https://en.wikipedia.org/wiki/Mollweide_projection
        theta = np.arcsin(y / np.sqrt(2))
        lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
        lon0 = 3 * np.pi / 2
        lon = lon0 + np.pi * x / (2 * np.sqrt(2) * np.cos(theta))

        # Add points at the poles
        lat = np.append(lat, [-np.pi / 2, np.pi / 2])
        lon = np.append(
            lon, [1.5 * np.pi, 1.5 * np.pi]
        )
        npix = len(lat)

        # Back to Cartesian, this time on the *sky*
        x = np.reshape(np.cos(lat) * np.sin(lon), [1, -1])
        y = np.reshape(np.sin(lat), [1, -1])
        z = np.reshape(np.cos(lat) * np.cos(lon), [1, -1])

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        # Flatten and fix the longitude offset, then sort by latitude
        lat = lat.reshape(-1)
        lon = (lon - 1.5 * np.pi).reshape(-1)
        idx = np.lexsort([lon, lat])
        lat = lat[idx]
        lon = lon[idx]
        x = x[idx]
        y = y[idx]
        z = z[idx]

    else:
        print("Unrecognized spatial sampling mode.")
        sys.exit()

    def calcflux(y):
        star, planet, system = initsystem(fit, lmax, y=y)
        lcfun = light_curves.light_curve(system, order=100)
        sflux, pflux = lcfun(t).T
        return sflux, pflux

    j_calcflux = jax.jit(calcflux)
    
    # Design matrix
    # Done manually. Would be nice if jaxoplanet had a function for this.
    Ny = (lmax+1)**2
    A = np.zeros((Nt, Ny))
    for i in range(Ny):
        yval = np.zeros(Ny + 1)
        
        yval[0] = 1.0
        yval[i] = 1.0

        sflux, A[:,i] = j_calcflux(yval)
        if i > 0:
            A[:,i] -= d.pflux_y00

    # Pixel transforms
    # Note: 'RV' is forced to False
    star, planet, system = initsystem(fit, lmax)
    pT = planet._poly_basis(False)(x, y, z)[:,:(lmax+1)**2]
    A1 = core.basis.A1(lmax)

    # Rotation - sky projection
    axis_x, axis_y, axis_z, angle = rotation.sky_projection_axis_angle(
        planet.inc, planet.obl)
    rotation_matrices = s2fft_rotation.compute_rotation_matrices(
        lmax, axis_x, axis_y, axis_z, angle)
    R1 = np.zeros((Ny, Ny))
    for l in range(lmax+1):
        R1[l*l:l*l+2*l+1,l*l:l*l+2*l+1] = rotation_matrices[l]

    # Rotation - inclination convention
    rotation_matrices = s2fft_rotation.compute_rotation_matrices(
        lmax, 1.0, 0.0, 0.0, -np.pi/2)
    R2 = np.zeros((Ny, Ny))
    for l in range(lmax+1):
        R2[l*l:l*l+2*l+1,l*l:l*l+2*l+1] = rotation_matrices[l]

    # Rotation - mystery 3rd rotation?? Jaxoplanet convention, maybe?
    # If you, dear reader, know why this is necessary, I'll buy you
    # a drink.
    rotation_matrices = s2fft_rotation.compute_rotation_matrices(
        lmax, 0.0, 1.0, 0.0, -np.pi)
    R3 = np.zeros((Ny, Ny))
    for l in range(lmax+1):
        R3[l*l:l*l+2*l+1,l*l:l*l+2*l+1] = rotation_matrices[l]
    #R3 = np.eye(Ny)

    Y2P = pT @ A1 @ R3 @ R2 @ R1

    lam = 1e-12
    P2Y = np.linalg.solve(Y2P.T.dot(Y2P) + lam * np.eye(Ny), Y2P.T)

    # Calculate visibility function
    vis = A @ P2Y

    # Get to the same units as the old visiblity function for simplicity
    vis /= np.pi

    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    
    return vis, lat, lon

def visibility_starry(fit, t, x, y, z, lmax):
    """
    Calculate the visibility of a grid of cells on a planet for
    a series of times using starry's design matrices to do this
    analytically. Returns the visibility array with the associated
    latitudes and longitudes. The sampling is done on a Mollweide
    grid such that all grid cells are of equal area.
    """
    # Lmax only influences the number of grid cells such that the
    # planet is well sampled.
    star, planet, system = initsystem(fit, lmax)

    rp = fit.cfg.planet.r
    rs = fit.cfg.star.r
    t0 = fit.cfg.planet.t0
    prot = fit.cfg.planet.prot

    rprs = rp / rs

    xo = (x[0] - x[1]) / rp
    yo = (y[0] - y[1]) / rp
    zo = (z[0] - z[1]) / rp

    # Rotation of the planet assuming 0 is mid-transit.
    theta = ((t - t0) / prot) % 1 * 360. + 180.

    A = planet.map.design_matrix(xo=xo, yo=yo, ro=1/rprs, theta=theta).eval()

    lat, lon, Y2P, P2Y, Dx, Dy = \
        planet.map.get_pixel_transforms(oversample=fit.cfg.threed.oversample,
                                        lam=1e-12)

    vis = A @ P2Y

    # Get to the same units as the old visiblity function for simplicity
    vis /= np.pi
    
    return vis, lat, lon
    
def visibility_old(t, latgrid, longrid, dlatgrid, dlongrid, theta0, prot,
                   t0, rp, rs, x, y):
    """
    Calculate the visibility of a grid of cells on a planet at a specific
    time. Returns a combined visibility based on the observer's
    line-of-sight, the area of the cells, and the effect of the star.

    Arguments
    ---------
    t: float
        Time to calculate visibility.
    
    latgrid: 2D array
        Array of latitudes, in radians, from -pi/2 to pi/2.

    longrid: 2D array
        Array of longitudes, in radians, from -pi to pi.

    dlat: float
        Latitude resolution in radians.

    dlon: float
        Longitude resoltuion in radians.

    theta0: float
        Rotation at t0 in radians.

    prot: float
        Rotation period, the same units as t.

    t0: float
        Time of transit, same units as t.

    rp: float
        Planet radius in solar radii.

    rs: float
        Star radius in solar radii.

    x: tuple
        x position of (star, planet)

    y: tuple
        y position of (star, planet)

    Returns
    -------
    vis: 2D array
        Visibility of each grid cell. Same shape as latgrid and longrid.

    """
    if latgrid.shape != longrid.shape:
        print("Number of latitudes and longitudes do not match.")
        raise Exception

    losvis  = np.zeros(latgrid.shape)
    starvis = np.zeros(latgrid.shape)
    
    # Flag to do star visibility calculation (improves efficiency)
    dostar = True

    # Central longitude (observer line-of-sight)
    centlon = theta0 - (t - t0) / prot * 2 * np.pi

    # Convert relative to substellar point
    centlon = (centlon + np.pi) % (2 * np.pi) - np.pi
    
    xsep = x[0] - x[1]
    ysep = y[0] - y[1]
    d = np.sqrt(xsep**2 + ysep**2)

    # Visible fraction due to star        
    # No grid cells visible. Return 0s
    if (d < rs - rp):
        return np.zeros(latgrid.shape)
    
    # All grid cells visible. No need to do star calculation.
    elif (d > rs + rp):
        starvis[:,:] = 1.0
        dostar     = False
    # Otherwise, time is during ingress/egress and we cannot simplify
    # calculation

    nlat, nlon = latgrid.shape
    for i in range(nlat):
        for j in range(nlon):
            # Angles wrt the observer
            lat  = latgrid[i,j]
            lon  = longrid[i,j]
            dlat = dlatgrid[i,j]
            dlon = dlongrid[i,j]
            
            phi   = lon - centlon
            theta = lat
            phimin   = phi - dlon / 2.
            phimax   = phi + dlon / 2.

            thetamin = lat - dlat / 2.
            thetamax = lat + dlat / 2.

            # Cell is not visible at this time. No need to calculate further.
            if (phimin > np.pi / 2.) or (phimax < -np.pi / 2.):
                losvis[i,j] = 0

            # Cell is visible at this time
            else:
                # Determine visible phi/theta range of the cell
                phirng   = np.array((np.max((phimin,   -np.pi / 2.)),
                                     np.min((phimax,    np.pi / 2.))))
                thetarng = np.array((np.max((thetamin, -np.pi / 2.)),
                                     np.min((thetamax,  np.pi / 2.))))


                # Visibility based on LoS
                # This is the integral of
                #
                # A(theta, phi) V(theta, phi) dtheta dphi
                #
                # where
                #
                # A = r**2 cos(theta)
                # V = cos(theta) cos(phi)
                #
                # Here we've normalized by pi*r**2, since
                # visibility will be applied to Fp/Fs where planet
                # size is already taken into account.
                losvis[i,j] = (np.diff(thetarng/2) + \
                               np.diff(np.sin(2*thetarng) / 4)) * \
                    np.diff(np.sin(phirng)) / \
                    np.pi

                # Grid cell maybe only partially visible
                if dostar:
                    thetamean = np.mean(thetarng)
                    phimean   = np.mean(phirng)
                    # Grid is "within" the star
                    if dgrid(x, y, rp, thetamean, phimean) < rs:
                        starvis[i,j] = 0.0
                    # Grid is not in the star
                    else:
                        starvis[i,j] = 1.0

    return starvis * losvis

def dgrid(x, y, rp, theta, phi):
    """
    Calculates the projected distance between a latitude (theta) and a 
    longitude (phi) on a planet with radius rp to a star. Projected
    star position is (x[0], y[0]) and planet position is (x[1], y[1]).
    """
    xgrid = x[1] + rp * np.cos(theta) * np.sin(phi)
    ygrid = y[1] + rp * np.sin(theta)
    d = np.sqrt((xgrid - x[0])**2 + (ygrid - y[0])**2)
    return d

def t_dgrid():
    """
    Returns a theano function of dgrid(), with the same arguments.
    """
    print('Defining theano function.')
    arg1 = theano.tensor.dvector('x')
    arg2 = theano.tensor.dvector('y')
    arg3 = theano.tensor.dscalar('rp')
    arg4 = theano.tensor.dscalar('theta')
    arg5 = theano.tensor.dscalar('phi')

    f = theano.function([arg1, arg2, arg3, arg4, arg5],
                        dgrid(arg1, arg2, arg3, arg4, arg5))    
    return f

def mapintensity(surface, lat, lon, amp):
    """
    Calculates a grid of intensities, multiplied by the amplitude given.
    """
     # Convert to radians (jaxoplanet expects radians)
    lat_rad = jnp.deg2rad(lat.flatten())
    lon_rad = jnp.deg2rad(lon.flatten())

      # Evaluate intensity at all positions
      # The Surface.intensity() method accepts arrays due to JAX broadcasting
    grid = surface.intensity(lat_rad, lon_rad)

      # Scale by amplitude
    grid = grid * amp

      # Reshape back to original shape
    grid = grid.reshape(lat.shape)

    return np.array(grid)


def hotspotloc_driver(fit, ln):
    """
    Calculates a distribution of hotspot locations based on the MCMC
    posterior distribution.

    Note that this function assumes the first ncurves parameters
    in the posterior are associated with eigencurves. This will not
    be true if some eigencurves are skipped over, as MC3 does not
    include fixed parameters in the posterior.

    Inputs
    ------
    fit: Fit instance

    map: Map instance (not starry Map)

    Returns
    -------
    hslocbest: tuple
        Best-fit hotspot location (lat, lon), in degrees.

    hslocstd: tuple
        Standard deviation of the hotspot location posterior distribution
        as (lat, lon)

    hspot: tuple
        Marginalized posterior distributions of latitude and longitude
    """
    
    post = ln.post[ln.zmask]

    nsamp, nfree = post.shape

    oversample =  5

    if fit.cfg.twod.ncalc > nsamp:
        print("Warning: ncalc reduced to match burned-in sample.")
        ncalc = nsamp
    else:
        ncalc = fit.cfg.twod.ncalc
    
    hslon = np.zeros(ncalc)
    hslat = np.zeros(ncalc)
    thinning = nsamp // ncalc

    def hotspotloc(y):
        star, planet, system = initsystem(fit, ln.lmax, y=y)
        (lat, lon), val = surface_is_physical.surface_min_intensity(
            planet, oversample, ln.lmax)

        return lat, lon, val

    j_hotspotloc = jax.jit(hotspotloc)

    # Note the maps created here do not include the correct uniform
    # component because that does not affect the location of the
    # hotspot. Also note that the eigenvalues are negated because
    # we want to maximize, not minimize, but jaxoplanet only includes
    # a minimize method.
    pbar = progressbar.ProgressBar(max_value=ncalc)
    for i in range(0, ncalc):
        ipost = i * thinning
        y = np.zeros((ln.lmax+1)**2)
        for j in range(ln.ncurves):
            y[1:] += -1 * post[ipost,j] * ln.eigeny[j,1:]
            y[0] = 1.0

        hslat[i], hslon[i], _ = j_hotspotloc(y)
        pbar.update(i+1)
        
    # Get the best-fit hotspot offset
    yb = np.zeros((ln.lmax+1)**2)
    yb[0] = 1.0
    for j in range(ln.ncurves):
        yb[1:] += -1 * ln.bestp[j] * ln.eigeny[j,1:]

    hslatbest, hslonbest, _ = hotspotloc(yb)

    # Convert to degrees
    hslat = np.rad2deg(hslat)
    hslon = np.rad2deg(hslon)
    hslatbest = np.rad2deg(hslatbest)
    hslonbest = np.rad2deg(hslonbest)

    # Constrain longitudes to [-180, 180]
    hslonbest = (hslonbest + 180.) % 360. - 180.
    hslon     = (hslon     + 180.) % 360. - 180.
    hslatbest = (hslatbest +  90.) % 180. -  90.
    hslat     = (hslat     +  90.) % 180. -  90.
    
    hslonstd = np.std(hslon)
    hslatstd = np.std(hslat)

    # Two-sided errors
    if ln.ncurves == 0:
        print("WARNING: Cannot determine hotspot location for uniform model.")
        print("         Do not trust hostspot location and uncertainties.")
        hsloncrlo = -180.
        hsloncrhi =  180.
        hslatcrlo =  -90.
        hslatcrhi =   90.
    else:
        pdf, xpdf, hpdmin = ms.cred_region(hslon)
        crlo = np.amin(xpdf[pdf>hpdmin])
        crhi = np.amax(xpdf[pdf>hpdmin])
        hsloncrlo = crlo - hslonbest
        hsloncrhi = crhi - hslonbest

        pdf, xpdf, hpdmin = ms.cred_region(hslat)
        crlo = np.amin(xpdf[pdf>hpdmin])
        crhi = np.amax(xpdf[pdf>hpdmin])
        hslatcrlo = crlo - hslatbest
        hslatcrhi = crhi - hslatbest

    hslocbest  = (hslatbest, hslonbest)
    hslocstd   = (hslatstd,  hslonstd)
    hslocpost  = (hslat,     hslon)
    hsloctserr = ((hslatcrhi, hslatcrlo), (hsloncrhi, hsloncrlo))
    
    return hslocbest, hslocstd, hslocpost, hsloctserr

def tmappost(fit, m, ln):
    post = ln.post[ln.zmask]

    nsamp, nfree = post.shape
    ncurves = ln.ncurves

    if fit.cfg.twod.ncalc > nsamp:
        print("Warning: ncalc reduced to match burned-in sample.")
        ncalc = nsamp
    else:
        ncalc = fit.cfg.twod.ncalc

    thinning = nsamp // ncalc

    fmaps = np.zeros((ncalc, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    tmaps = np.zeros((ncalc, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    
    star, planet, system = initsystem(fit, ln.lmax)

    nloc = len(fit.lat.flatten())

    lat = fit.lat.flatten()
    lon = fit.lon.flatten()

    # Lil function to calculate a single flux map. Returns a flattened
    # array that needs to be reshaped into 2D
    # (Possibly could be replaced with eigen.mkmaps()? Or move this
    #  function to be accessible elsewhere?)
    def calcfmap(yval: jnp.array,
                 unifamp: float):
        star, planet, system = initsystem(fit, ln.lmax, yval)
        
        fmap = planet.intensity(np.deg2rad(lat),
                                np.deg2rad(lon))

        # Fitted uniform component (-1 to remove default uniform
        # component). We could calculate this with a jaxoplanet object,
        # but it's faster to just use the knowledge that a uniform
        # map has Y00 / pi intensity everywhere.
        fmap += (unifamp - 1) / np.pi

        return fmap

    j_calcfmap = jax.jit(calcfmap)
    
    pbar = progressbar.ProgressBar(max_value=ncalc)
    for i in range(ncalc):
        ipost = i * thinning
        
        yval = np.zeros((ln.lmax+1)**2)
        yval[0] = 1.0
        
        for j in range(ln.ncurves):
            yval[1:] += post[ipost,j] * ln.eigeny[j,1:]

        yval = jnp.array(yval)
        fmaps[i] = j_calcfmap(yval, post[ipost, ncurves]).reshape(fit.lat.shape)
        tmaps[i] = fmap_to_tmap(fmaps[i], m.wlmid, fit.cfg.planet.r,
                                fit.cfg.star.r, fit.cfg.star.t,
                                post[ipost,ncurves+1],
                                starspec=fit.cfg.star.starspec,
                                fwl=m.filtwl, ftrans=m.filttrans,
                                swl=fit.starwl, sspec=fit.starflux,
                                trange=m.trange,
                                fpfs_bb=m.fpfs_for_interp)
        
        pbar.update(i+1)

    return fmaps, tmaps

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

def ess(chain):
    '''
    Calculates the Steps Per Effectively-Independent Sample and
    Effective Sample Size (ESS) of a chain from an MCMC posterior 
    distribution.

    Adapted from some code I wrote for MC3 many years ago, and
    the SPEIS/ESS calculation in BART.
    '''
    nciter, npar = chain.shape

    speis = np.zeros(npar)
    ess   = np.zeros(npar)

    for i in range(npar):
        mean     = np.mean(chain[:,i])
        autocorr = np.correlate(chain[:,i] - mean,
                                chain[:,i] - mean,
                                mode='full')
        # Keep lags >= 0 and normalize
        autocorr = autocorr[np.size(autocorr) // 2:] / np.max(autocorr)
        # Sum adjacent pairs (Geyer, 1993)
        pairsum = autocorr[:-1:2] + autocorr[1::2]
        # Find where the sum goes negative, or use the whole thing
        if np.any(pairsum < 0):
            idx = np.where(pairsum < 0)[0][0]
        else:
            idx = len(pairsum)
            # Only warn the user if this parameter was varied
            if len(np.unique(chain[:,i])) > 1:
                print("WARNING: parameter {} did not decorrelate!"
                      "Do not trust ESS/SPEIS!".format(i))
        # Calculate SPEIS
        speis[i] = -1 + 2 * np.sum(pairsum[:idx])
        ess[i]   = nciter / speis[i]

    return speis, ess

def crsig(ess, cr=0.683):
    '''
    Calculates the absolute error on an estimate of a credible region
    of a given percentile based on the effective sample size.

    See Harrington et al, 2021.

    Arguments
    ---------
    ess: int
        Effective Sample Size

    cr: float
        Credible region percentile to calculate error on. E.g., 
        for a 1-sigma region, use 0.683 (the default).

    Returns
    -------
    crsig: float
        The absolute error on the supplied credible region.
    '''
    return (cr * (1 - cr) / (ess + 3))**0.5

@njit
def fast_linear_interp(a, b, x):
    return (b[1] - a[1]) / (b[0] - a[0]) * (x - a[0]) + a[1]

def blackbody(T, wn):
    '''
    Calculates the Planck function for a grid of temperatures and
    wavenumbers. Wavenumbers must be in /cm.
    '''    
    # Convert from /cm to /m
    wn_m = wn * 1e2
    bb = (2.0 * sc.h * sc.c**2 * wn_m[np.newaxis]**3) \
        * 1 / (np.exp(sc.h * sc.c * wn_m[np.newaxis] / sc.k / T[:, np.newaxis]) - 1.0)

    return bb

def blackbody_wl(T, wl):
    '''
    Calculates the Planck function for a grid of temperatures and
    wavelengths. Wavelenghts must be in m.
    '''
    bb = (2.0 * sc.h * sc.c**2 / (wl[np.newaxis]**5)) \
        * 1 / (np.exp(sc.h * sc.c / wl[np.newaxis] / sc.k / T[:, np.newaxis]) - 1.0)
    
    return bb
    
