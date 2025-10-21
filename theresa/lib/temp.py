import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.orbit import SurfaceSystem, SurfaceBody
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.light_curves import light_curve
from lib import pca
from lib import dummyfit

def initsystem(fit, ydeg):
    '''
    Uses a fit object to build the respective starry objects. Useful
    because starry objects cannot be pickled. Returns a tuple of
    (star, planet, system).
    '''
    
    cfg = fit.cfg
    star_ylm = Ylm.from_dense(jnp.array([1.0]), normalize=True)

    star_surface = Surface(
          y=star_ylm,
          inc=jnp.pi/2,              # Edge-on inclination
          period=cfg.star.prot,       # Rotation period in days
          radius=cfg.star.r,          # Radius in solar radii
          amplitude=1.0               # Normalized amplitude
      )

      # Create planet surface with spherical harmonics up to ydeg
      # Initialize all coefficients to zero except Y_00 = 1.0 (uniform map)
    n_coeffs = (ydeg + 1)**2
    planet_ylm_coeffs = jnp.zeros(n_coeffs)
    planet_ylm_coeffs = planet_ylm_coeffs.at[0].set(1.0)  # Y_00 = 1.0
    planet_ylm = Ylm.from_dense(planet_ylm_coeffs, normalize=True)

    planet_surface = Surface(
        y=planet_ylm,
        inc=jnp.deg2rad(cfg.planet.inc),     # Inclination in radians
        period=cfg.planet.prot,               # Rotation period in days
        radius=cfg.planet.r,                  # Radius in solar radii
        amplitude=1.0,
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

    return star_surface, planet_surface, system


def vislon(system, data):
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

      return float(jnp.min(limb1)), float(jnp.max(limb2))

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

    # Helper function to create a new system with modified planet surface
    def create_system_with_surface(new_planet_surface, time_transit_offset=0.0):
        """Create a new system with updated planet surface."""
        # Just modify the body_surfaces tuple
        if time_transit_offset == 0.0:
            # Simple case - just replace the surface
            return system.__class__(
                central=system.central,
                central_surface=system.central_surface,
                bodies=system.bodies,
                body_surfaces=(new_planet_surface,)
            )
        else:
            # Need to modify time_transit - recreate the body
            # This requires knowing orbital parameters, handle separately
            return None  # Will handle in orbcheck section

    # Jaxoplanet function to evaluate flux with modified Ylm coefficients
    def evalflux(yval, base_system=None):
        """
        Compute light curve for given Ylm coefficients.
        yval is array of coefficients (excluding Y00).
        Returns star flux and planet flux separately.
        """
        if base_system is None:
            base_system = system

        # Create full Ylm coefficient array (including Y00 = 1)
        n_coeffs = (lmax + 1)**2
        ylm_coeffs = jnp.zeros(n_coeffs)
        ylm_coeffs = ylm_coeffs.at[0].set(1.0)  # Y00 = 1 (uniform)
        ylm_coeffs = ylm_coeffs.at[1:].set(yval)  # Higher order terms

        # Create new Ylm with these coefficients
        new_ylm = Ylm.from_dense(ylm_coeffs, normalize=False)

        # Create new planet surface with updated Ylm
        new_planet_surface = Surface(
            y=new_ylm,
            inc=planet_surface.inc,
            period=planet_surface.period,
            radius=planet_surface.radius,
            amplitude=planet_surface.amplitude,
            phase=planet_surface.phase
        )

        # Create new system with updated planet surface using eqx.tree_at
        # Replace the body_surfaces tuple with new surface
        new_system = eqx.tree_at(
            lambda sys: sys.body_surfaces,
            base_system,
            (new_planet_surface,)
        )

        # Compute light curve - returns array with shape (n_bodies, n_times)
        flux_result = light_curve(new_system, order=20)(t)

        # Extract star and planet fluxes
        # flux_result.T[0] is star, flux_result.T[1] is planet
        starflux = np.array(flux_result.T[0])
        planetflux = np.array(flux_result.T[1])

        return starflux, planetflux

    # Create harmonic maps of the planet, excluding Y00
    # (lmax**2 maps, plus a negative version for all but Y00)
    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ilc = 0
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            # Create array of Ylm coefficients (excluding Y00)
            yval = np.zeros(nharm // 2)

            # Set this specific harmonic to +1.0
            yval[ilc // 2] = 1.0
            sflux, lcs[ilc] = evalflux(yval)

            # Set this specific harmonic to -1.0
            yval[ilc // 2] = -1.0
            sflux, lcs[ilc+1] = evalflux(yval)

            ilc += 2

    # If user wants to include additional eigencurves which explore
    # different orbital parameters
    if orbcheck is not None:
        # TODO: Implement orbcheck for jaxoplanet
        # For now, skip this feature
        print("Warning: orbcheck not yet implemented for jaxoplanet, skipping...")

    # Subtract uniform map contribution (jaxoplanet includes this in all light curves)
    lcs -= y00
            
    # Run PCA to determine orthogonal light curves
    if ncurves is None:
        ncurves = nharm
        if method == 'tsvd':
            ncurves -= 1
        
    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)

    # Convert orthogonal light curves into maps        
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

    return eigeny, evalues, evectors, proj, lcs


def main():
    fit = dummyfit.create_dummy_fit()

    star_surface, planet_surface, system = initsystem(fit, 3)
    #print(system.t)



    

main()