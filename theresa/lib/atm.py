import os
import sys
import numpy as np
import scipy as sp
import scipy.constants as sc
import constants as c
import utils
import scipy.interpolate as spi
import time

libdir = os.path.dirname(os.path.realpath(__file__))
moddir = os.path.join(libdir, 'modules')
ratedir = os.path.join(moddir, 'rate')

sys.path.append(ratedir)
import rate

def atminit(atmtype, mols, p, t, mp, rp, refpress, elemfile, outdir,
            ilat=None, ilon=None, cheminfo=None):
    """
    Initializes atmospheres of various types.
    
    Inputs
    ------
    atmtype: string
        Type of atmosphere to initialize. Options are:
            rate: thermochemical eqilibrium with RATE
            ggchem: thermochemical equilibrium with GGchem (requires file)

    p: 1D array
        Pressure layers of the atmosphere

    t: 3D array
        Temperature array, of size (nlayers, res, res)
    
    mp: float
        Mass of the planet, in solar masses

    rp: float
        Radius of the planet, in solar radii

    refpress: float
        Reference pressure at rp (i.e., p(rp) = refpress). Used to calculate
        radii of each layer, assuming hydrostatic equilibrium.

    elemfile: string
        File containing elemental molar mass information. See 
        inputs/abundances_Asplund2009.txt for format.

    outdir: string
        Directory where atmospheric file will be written.

    ilat: 2d array
        Optional array of latitude indices where atmosphere should 
        be evaluated.

    ilon: 2d array
        Optional array of longitude indices where atmosphere should
        be evaluated.

    cheminfo: list or tuple
        Iterable that contains information needed by certain
        atmtypes. For example, GGchem requires temperatures,
        pressures, species, and abundances from a preloaded
        file.

    Returns
    -------
    r: 1D array
        Radius at each layer of the atmosphere.

    p: 1D array
        Pressure at each layer of the atmosphere.

    abn: 2D array
        Abundance (mixing ratio) of each species in the atmosphere.
        Rows are atmosphere layers and columns are species abundances.
    """

    # Convert planet mass and radius to Jupiter
    rp *= c.Rsun / c.Rjup
    mp *= c.Msun / c.Mjup

    nlayers, nlat, nlon = t.shape

    if ilat is None:
        ilat = np.repeat(np.arange(nlat), nlon)
    if ilon is None:
        ilon = np.repeat(np.arange(nlon), nlat)
    
    # Equilibrium atmosphere
    if atmtype == 'rate':
        robj = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
        spec = robj.species
        nspec = len(spec)
        abn = np.zeros((nspec, nlayers, nlat, nlon))
        for i, j in zip(ilat, ilon):
            abn[:,:,i,j] = robj.solve(t[:,i,j], p)

    elif atmtype == 'ggchem':
        ggchemT, ggchemp, spec, ggchemabn = cheminfo
        tic = time.time()
        ngrid, nspec = ggchemabn.shape
        abn = np.zeros((nspec, nlayers, nlat, nlon))
        
        if not np.all(np.isclose(p, np.sort(np.unique(ggchemp))[::-1])):
            print("Pressures of fit and atmosphere file do not match. Exiting")
            sys.exit()

        for s in range(nspec):
            if spec[s] in mols:
                for k in range(nlayers):
                    where = np.where(np.isclose(p[k], ggchemp))
                    interpT   = ggchemT[where]
                    interpabn = ggchemabn[where,s][0]
                    tsorter = np.argsort(interpT)
                    for i, j in zip(ilat, ilon):
                        idx1 = np.searchsorted(interpT[tsorter], t[k,i,j],
                                               side='left')
                        idx2 = idx1 - 1
                        abn[s,k,i,j] = utils.fast_linear_interp(
                            (interpT[tsorter][idx1],
                             interpabn[tsorter][idx1]),
                            (interpT[tsorter][idx2],
                             interpabn[tsorter][idx2]),
                            t[k,i,j])

    else:
        print("Unrecognized atmopsphere type.")
        sys.exit()

    return abn, spec

def atmsave(r, p, t, abn, spec, outdir, atmfile):
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
                        abn))

    with open(os.path.join(outdir, atmfile), 'w') as f:
        f.write('# Atmospheric File\n')
        f.write('ur {}\n'.format(1e2))
        f.write('up {}\n'.format(1e6))
        f.write('q number\n')
        f.write('#SPECIES\n')
        f.write(' '.join(spec) + '\n')
        f.write('#TEADATA\n')

    with open(os.path.join(outdir, atmfile), 'a') as f:
        np.savetxt(f, atmarr, fmt='%.4e')
        
        
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
    with open(atmfile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Take out comments
        if line.startswith('#') and not \
           line.startswith('#SPECIES') and not \
           line.startswith('#TEADATA'):
            lines.remove(line)
        # Take out blank or whitespace lines
        if not line.rstrip():
            lines.remove(line)

    for i, line in enumerate(lines):
        if line.startswith('ur'):
            ur = float(line.split()[-1])
        if line.startswith('up'):
            up = float(line.split()[-1])
        if line.startswith('q'):
            q  = line.split()[-1]
        if line.startswith('#SPECIES'):
            ispec = i + 1
        if line.startswith('#TEADATA'):
            idata = i + 1

    spec = lines[ispec].split()

    datalines = lines[idata:]

    nlayer = len(datalines)
    ncol   = len(datalines[0].rstrip().split())

    arr = np.zeros((nlayer, ncol))
    
    for i in range(nlayer):
        arr[i] = np.array(datalines[i].rstrip().split(), dtype=float)

    r   = arr[:,0 ]
    p   = arr[:,1 ]
    t   = arr[:,2 ]
    abn = arr[:,3:]

    return r, p, t, abn, spec
                        
def calcmu(elemfile, abn, spec):
    """
    Calculates the mean molar mass of each layer of an atmosphere.

    Arguments
    ---------
    elemfile: string
        File containing elemental mass information.

    abn: 2D array
        Array of atmospheric abundances. Rows are layers, columns are
        species.

    spec: list
        List of strings of species in the atmopshere corresponding to
        the columns of the abn array.

    Returns
    -------
    mu: 1D array
        Mean molecular mass, in g/mol, of each layer of the atmosphere.
    """

    dtype = [('idx', int), ('elem', 'U2'), ('dex', float),
             ('name', 'U10'), ('mass', float)]
    elemarr = np.genfromtxt(elemfile, comments='#',
                            dtype=dtype)

    idx  = elemarr['idx']
    elem = elemarr['elem']
    dex  = elemarr['dex']
    name = elemarr['name']
    mass = elemarr['mass'] 

    nspec, nlayer = abn.shape

    mu = np.zeros(nlayer)

    specweight = np.zeros(nspec)

    for i in range(nspec):
        specelem, specnum = mol_to_elem(spec[i])
        for j in range(len(specelem)):
            elemidx = np.where(specelem[j] == elem)[0]
            specweight[i] += mass[elemidx] * specnum[j]

    for i in range(nlayer):
        mu[i] = np.sum(specweight * abn[:,i])

    return mu

def mol_to_elem(mol):
    elem = []
    num  = []

    # Catch for single-character elements
    if len(mol) == 1:
        elem.append(mol)
        num.append(1)
        return elem, num

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
    """
    Calculates the radius of each layer of an atmosphere, given
    pressure, temperature, mean molecular weight, planet radius,
    planet mass, and pressure at the planet radius.

    Arguments
    ---------
    p: 1D array
        Pressure array (bars)

    t: 1D array
        Temperature array (K)

    mu: 1D array
        Mean molecular mass (g/mol)

    r0: float
        Planetary radius (Rjup)

    mp: float
        Planetary mass (Mjup) 

    p0: float
        Pressure at r0 (bars)

    Returns
    -------
    r: 1D array
        Radius of each layer in meters

    """
    nlayer = len(p)
    r = np.zeros(nlayer)
    g = np.zeros(nlayer)

    # Convert mu to kg/mol
    mu /= 1000

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
        r[i0] = r0
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
        

def tgrid(nlayers, nlat, nlon, tmaps, pmaps, pbot, ptop, params,
          interptype='linear', oob='extrapolate', smooth=None):
    """
    Make a 3d grid of temperatures, based on supplied temperature maps
    placed at the supplied pressures. Dimensions are (nlayers, nlat,
    nlon). Will optionally smooth with a rolling average.

    """
    
    temp3d = np.zeros((nlayers, nlat, nlon))

    logp1d = np.linspace(np.log10(pbot), np.log10(ptop), nlayers)

    for i in range(nlat):
        for j in range(nlon):
            if oob == 'extrapolate':
                fill_value = 'extrapolate'
                p_interp = np.copy(pmaps[:,i,j])
                t_interp = np.copy(tmaps[:,i,j])
            elif oob == 'isothermal':
                imax = np.argsort(pmaps[:,i,j])[-1]
                imin = np.argsort(pmaps[:,i,j])[0]
                fill_value = (tmaps[:,i,j][imin], tmaps[:,i,j][imax])
                p_interp = np.copy(pmaps[:,i,j])
                t_interp = np.copy(tmaps[:,i,j])
            elif oob == 'top':
                ttop = params[-1]
                p_interp = np.concatenate((pmaps[:,i,j], (ptop,)))
                t_interp = np.concatenate((tmaps[:,i,j], (ttop,)))
                imax = np.argsort(pmaps[:,i,j])[-1]
                fill_value = (ttop, tmaps[:,i,j][imax])
            elif oob == 'bot':
                tbot = params[-1]
                p_interp = np.concatenate((pmaps[:,i,j], (pbot,)))
                t_interp = np.concatenate((tmaps[:,i,j], (tbot,)))
                imin = np.argsort(pmaps[:,i,j])[0]
                fill_value = (tmaps[:,i,j][imin], tbot)
            elif oob == 'both':
                ttop = params[-2]
                tbot = params[-1]
                p_interp = np.concatenate((pmaps[:,i,j],
                                           (ptop, pbot)))
                t_interp = np.concatenate((tmaps[:,i,j],
                                           (ttop, tbot)))
                fill_value = 'extrapolate' # shouldn't matter
                
            interp = spi.interp1d(np.log10(p_interp),
                                  t_interp, kind=interptype,
                                  bounds_error=False,
                                  fill_value=fill_value)
            
            temp3d[:,i,j] = interp(logp1d)

            if smooth is not None:
                T = temp3d[:,i,j]
                Tsmooth = np.convolve(T, np.ones(smooth), 'valid') / smooth
                nedge = np.int((len(T) - len(Tsmooth)) / 2)
                temp3d[:,i,j][nedge:-nedge] = Tsmooth
            
    return temp3d, 10**logp1d

def pmaps(params, fit):
    '''
    Calculates pressures of tmaps using a variety of mapping functions.
    '''
    tmaps = fit.tmaps
    lat   = fit.lat
    lon   = fit.lon
    dlat  = fit.dlat
    dlon  = fit.dlon
    mapfunc = fit.cfg.threed.mapfunc
    
    pmaps = np.zeros(tmaps.shape)
    nmap, nlat, nlon = pmaps.shape
    if   mapfunc == 'isobaric':
        for i in range(nmap):
            pmaps[i] = 10.**params[i]
    elif mapfunc == 'sinusoidal':
        npar = 4
        for i in range(nmap):
            ip = npar * i
            pmaps[i] = 10.**(params[ip] + \
                params[ip+1]*np.cos((lat             )*np.pi/180.) + \
                params[ip+2]*np.cos((lon-params[ip+3])*np.pi/180.))
    elif mapfunc == 'flexible':
        ilat, ilon = np.where((lon + dlon / 2. > fit.minvislon) &
                              (lon - dlon / 2. < fit.maxvislon))
        nvis = len(ilat)
        for i in range(nmap):
            ip = 0
            for j, k in zip(ilat, ilon):
                pmaps[i,j,k] = 10.**params[i*nvis+ip]
                ip += 1
    elif mapfunc == 'quadratic':
        npar = 6
        for i in range(nmap):
            ip = npar*i
            pmaps[i] = 10.**(
                params[ip  ]          +
                params[ip+1] * lat**2 +
                params[ip+2] * lon**2 +
                params[ip+3] * lat    +
                params[ip+4] * lon    +
                params[ip+5] * lat*lon)
    elif mapfunc == 'cubic':
        npar = 10
        for i in range(nmap):
            pmaps[i] = 10.**(
                params[ip  ]              +
                params[ip+1] * lat**3     +
                params[ip+2] * lon**3     +
                params[ip+3] * lat**2     +
                params[ip+4] * lon**2     +
                params[ip+5] * lat        +
                params[ip+6] * lon        +
                params[ip+7] * lat**2*lon +
                params[ip+8] * lat*lon**2 +
                params[ip+9] * lat*lon)
    else:
        print("WARNING: Unrecognized pmap model.")

    return pmaps
        
def read_GGchem(fname):
    '''
    Read a GGchem output file.

    Inputs
    ------
    fname: string
        File to be read.

    Returns
    -------
    T: 1D array
        Array of temperatures
   
    p: 1D array
        Array of pressures

    spec: list of strings
        Elemental and molecular species names

    abn: 2D array
        Elemental and molecular number mixing ratios (same as molar
        mixing ratios)
    '''
    
    with open(fname) as f:
        f.readline() # skip first line
        d = np.array(f.readline().split())
        nelem = int(d[0])
        nmol  = int(d[1])
        ndust = int(d[2])
        npt   = int(d[3])
        header = f.readline().split()

    data = np.loadtxt(fname, skiprows=3)

    T = data[:,0]
    p = data[:,2] / 1e6 # convert to bars
    spec = header[3:4+nelem+nmol]

    ntot = 0
    for i in range(3, 4 + nelem + nmol):
        ntot += 10**data[:,i]

    # Mixing ratio in log space (calculated by molecular number, but
    # same as volume mxing ratio)
    abn = data[:,3:4+nelem+nmol] - np.log10(ntot).reshape(npt**2, 1)

    # Convert to non-log space
    abn = 10.**abn

    return T, p, spec, abn

    
        
        
        
    
    
        
