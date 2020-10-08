import numpy as np
import pickle
import scipy.interpolate as spi

def specint(wn, spec, filtfiles):
    """
    Integrate a spectrum over the given filters.

    Arguments
    ---------
    wn: 1D array
        Wavenumbers (/cm) of the spectrum

    spec: 1D array
        Spectrum to be integrated

    filtfiles: list
        Paths to filter files, which should be 2 columns.
        First column is wavelength (microns), and second column
        is filter throughput.

    Returns
    -------
    intspec: 1D array
        The spectrum integrated over each filter. 
    """
    intspec = np.zeros(len(filtfiles)) 
    
    for i, filtfile in enumerate(filtfiles):
        filtwl, trans = np.loadtxt(filtfile, unpack=True)

        um2cm = 1e-4
        
        filtwn = 1.0 / (filtwl * um2cm)

        # Sort ascending
        idx = np.argsort(filtwn)
        filtwn = filtwn[idx]
        trans  = trans[idx]
        
        intfunc = spi.interp1d(filtwn, trans, bounds_error=False,
                               fill_value=0)

        # Interpolate transmission
        inttrans = intfunc(wn)

        # Normalize to one
        norminttrans = inttrans / np.trapz(inttrans, wn)

        # Integrate filtered spectrum
        intspec[i] = np.trapz(spec * norminttrans, wn)

    return intspec

    
def vislon(planet, fit):
    """
    Determines the range of visible longitudes based on times of
    observation.

    Arguments
    ---------
    planet: starry Planet object
        Planet object

    fit: Fit object
        Fit object. Must contain observation information.

    Returns
    -------
    minlon: float
        Minimum visible longitude, in degrees

    maxlon: float
        Maximum visible longitude, in degrees
    """
    t = fit.t

    porb   = planet.porb   # days / orbit
    prot   = planet.prot   # days / rotation
    t0     = planet.t0     # days
    theta0 = planet.theta0 # degrees

    # Central longitude at each time ("sub-observer" point)
    centlon = theta0 - (t - t0) / prot * 360

    # Minimum and maximum longitudes (assuming +/- 90 degree
    # visibility)
    limb1 = centlon - 90
    limb2 = centlon + 90

    # Rescale to [-180, 180]
    limb1 = (limb1 + 180) % 360 - 180
    limb2 = (limb2 + 180) % 360 - 180

    return np.min(limb1.eval()), np.max(limb2.eval())
    
    
    
