"""
2D mapping model functions - extracted from model.py to avoid theano dependencies.
These functions are used for 2D phase curve mapping and do not require theano/starry.
"""

import numpy as np
from numba import jit, literal_unroll


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
    for dvec in literal_unroll(dvecs):
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
            params   = np.concatenate((params,   (1.0, 0.00001, 0.00001, 0.00001)))
            pstep    = np.concatenate((pstep,    (0.01, 0.01, 0.01,    0.01)))
            pmin     = np.concatenate((pmin,     (0.8, -5,  -5, -5)))
            pmax     = np.concatenate((pmax,     (1.2, 30, 30,  30)))
            pnames   = np.concatenate((pnames,   ('r0', 'r1', 'r2', 'r3')))
            texnames = np.concatenate((texnames, ('$r_0$', '$r_1$', '$r_2$', '$r_3$')))
            npar = 4
        else:
            print("Baseline model {} not recognized.".format(v.baseline))

        nramppar.append(npar)

    # pindex is used to grab only the necessary parameters
    # for each model
    npars = [nmappar, nnormpar] + ndvecpar + nramppar
    nparams = len(params)
    pindex = np.zeros((len(npars), nparams))
    for i, npar in enumerate(npars):
        if i == 0:
            start = 0
        else:
            start = np.sum(npars[:i])
        pindex[i, start:start+npar] = True

    pindex = pindex.astype(bool)

    return params, pstep, pmin, pmax, pnames, texnames, pindex
