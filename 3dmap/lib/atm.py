import os
import sys
import numpy as np
import scipy as sp
import scipy.constants as sc
import constants as c

libdir = os.path.dirname(os.path.realpath(__file__))
moddir = os.path.join(libdir, 'modules')
ratedir = os.path.join(moddir, 'rate')

sys.path.append(ratedir)
import rate

def atminit(atmtype, atmfile, nlayers, ptop, pbot, t, outdir, rp, refpress,
            elemfile):
    if type(t) == float or type(t) == int:
        t = np.ones(nlayers) * t

    if os.path.isfile(atmfile):
        print("Using atmosphere " + atm)
        r, p, t, abn = atmload(atmfile)
        atmsave(r, p, t, abn, outdir, atmfile)
        return r, p, t, abn

    p = np.logspace(pbot, ptop, nlayers)

    # Equilibrium atmosphere
    if atmtype == 'eq':
        robj = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
        abn  = robj.solve(t, p).T
        spec = robj.species 

    mu = calcmu(elemfile, abn, spec)

    r  = calcrad(p, t, mu, rp, refpress)
        
    atmsave(r, p, t, abn, outdir, atmfile)
    return r, p, t, abn

def atmsave(r, p, t, abn, outdir, atmfile):
    """
    Save an atmosphere file. Columns are pressure, temeprature, and abundance.

    Inputs:
    -------
    r: 1D array
        Radius array

    p: 1D array
        Pressure array

    t: 1D array
        Temperature array

    abn: 1D array
        Abundance array. Rows are abundances for each molecule, and
        columns are pressure.

    """
    nlayers = len(p)
    atmarr = np.hstack((r.reshape((nlayers, 1)),
                        p.reshape((nlayers, 1)),
                        t.reshape((nlayers, 1)),
                        abn.T))
    np.savetxt(os.path.join(outdir, atmfile), atmarr,
               fmt='%.4e')

def atmload(atmfile):
    """
    Load an atmosphere file.

    Inputs:
    -------
    atmfile: str
        File to load.

    Returns:
    --------
    r: 1D array
        Radius array

    p: 1D array
        Pressure array

    t: 1D array
        Temperature array

    abn: 1D array
        Abundance array. Rows are abundances for each molecule, and
        columns are pressure.

    """
    arr = np.loadtxt(atmfile)

    r   = arr[:,0 ]
    p   = arr[:,1 ]
    t   = arr[:,2 ]
    abn = arr[:,3:]

    return p, t, abn
                        
def calcmu(elemfile, abn, spec):

    dtype = [('idx', int), ('elem', 'U2'), ('dex', float),
             ('name', 'U10'), ('mass', float)]
    elemarr = np.genfromtxt(elemfile, comments='#',
                            dtype=dtype)

    idx  = elemarr['idx']
    elem = elemarr['elem']
    dex  = elemarr['dex']
    name = elemarr['name']
    mass = elemarr['mass']

    nlayer, nspec = abn.shape

    mu = np.zeros(nlayer)

    specweight = np.zeros(nspec)

    for i in range(nspec):
        specelem, specnum = mol_to_elem(spec[i])
        for j in range(len(specelem)):
            elemidx = np.where(specelem[j] == elem)[0]
            specweight[i] += mass[elemidx] * specnum[j]

    for i in range(nlayer):
        mu[i] = np.sum(specweight * abn[i])

    return mu

def mol_to_elem(mol):
    elem = []
    num  = []

    start = 0

    # Start from idx 1 since we can assume the first letter is the beginning
    # of an element
    for i in range(1, len(mol)):
        # Reached the end of the element
        if mol[i].isupper():
            # If more than one of the element
            if mol[i-1].isdigit():
                elem.append(mol[start:i-1])
                num.append(int(mol[i-1]))
            # If only one atom
            else:
                elem.append(mol[start:i])
                num.append(1)
            start = i
        # Special case end-of-string
        if i == len(mol) - 1:
            if mol[i].isdigit():
                elem.append(mol[start:i])
                num.append(int(mol[i]))
            else:
                elem.append(mol[start:])
                num.append(1)
                

    return elem, num

def calcrad(p, t, mu, r0, mp, p0):
    nlayer = len(p)
    r = np.zeros(nlayer)
    g = np.zeros(nlayer)

    interpt  = sp.interpolate.interp1d(np.log10(p), t)
    interpmu = sp.interpolate.interp1d(np.log10(p), mu)
            
    t0  = interpt( np.log10(p0))
    mu0 = interpmu(np.log10(p0))

    mp *= c.Mjup
    r0 *= c.Rjup
    
    g0 = sc.G * mp / r0**2

    i0 = np.argmin(np.abs(p - p0))

    if p[i0] != p0:
        sgn = [-1, 1][p[i0] > p0]
        r[i0] = r0 + sgn * 0.5 * (t[i0] / mu[i0] + t0 / mu0) \
                * (sc.Avogadro * sc.k * sgn * np.log(p0/p[i0]) / g0)
        g[i0] = g0 * r0**2 / r[i0]**2
    else:
        r[i0] = rp
        g[i0] = g0

    # Calculate out from r0
    for i in range(i0+1, nlayer):
        r[i] = r[i-1] + 0.5 * (t[i] / mu[i] + t[i-1] / mu[i-1]) \
               * (sc.Avogadro * sc.k * np.log(p[i-1] / p[i]) / g[i-1])
        g[i] = g[i-1] * r[i-1]**2 / r[i]**2
    for i in range(i0-1, -1, -1):
        r[i] = r[i+1] - 0.5 * (t[i] / mu[i] + t[i+1] / mu[i+1]) \
               * (sc.Avogadro * sc.k * np.log(p[i] / p[i+1]) / g[i+1])
        g[i] = g[i+1] * r[i+1]**2 / r[i]**2

    return r
        

    
    
        

    
    
        
