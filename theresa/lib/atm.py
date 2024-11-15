import os
import sys
import numpy as np
import scipy as sp
import scipy.constants as sc
import constants as c
import utils
import scipy.interpolate as spi
import time
import taurex_ggchem
import progressbar

libdir = os.path.dirname(os.path.realpath(__file__))
moddir = os.path.join(libdir, 'modules')
ratedir = os.path.join(moddir, 'rate')

sys.path.append(ratedir)
import rate

def atminit(atmtype, mols, p, t, mp, rp, refpress, z,
            ivis=None, cheminfo=None):
    """
    Initializes atmospheres of various types.
    
    Inputs
    ------
    atmtype: string
        Type of atmosphere to initialize. Options are:
            rate: thermochemical eqilibrium with RATE
            ggchem: thermochemical equilibrium with GGchem

    p: 1D array
        Pressure layers of the atmosphere

    t: 2D array
        Temperature array, of size (nlayers, ncolumns)
    
    mp: float
        Mass of the planet, in solar masses

    rp: float
        Radius of the planet, in solar radii

    refpress: float
        Reference pressure at rp (i.e., p(rp) = refpress). Used to calculate
        radii of each layer, assuming hydrostatic equilibrium.

    outdir: string
        Directory where atmospheric file will be written.

    z: float
        Metallicity. E.g., z=0 is solar.

    ivis: 1d array
        Optional array of indices where atmosphere should 
        be evaluated.

    cheminfo: list or tuple
        Iterable that contains information needed by certain
        atmtypes. For example, GGchem requires temperatures,
        pressures, species, and abundances from a preloaded
        file.

    Returns
    -------
    abn: 3D array
        Abundance (mixing ratio) of each species in the atmosphere.
        nspec X nlayer X ncolumn

    spec: list
        Species associated with the abundances in abn
    """

    # Convert planet mass and radius to Jupiter
    rp *= c.Rsun / c.Rjup
    mp *= c.Msun / c.Mjup

    nlayer, ncolumn = t.shape

    if ivis is None:
        ivis = np.arange(ncolumn)
    
    # Equilibrium atmosphere
    if atmtype == 'rate':
        robj = rate.Rate(C=2.5e-4*(10**z),
                         N=1.0e-4*(10**z),
                         O=5.0e-4*(10**z),
                         fHe=0.0851)
        spec = robj.species
        nspec = len(spec)
        abn = np.zeros((nspec, nlayer, ncolumn))
        for i in ivis:
            abn[:,:,i] = robj.solve(t[:,i], p)

    elif atmtype == 'ggchem':
        ggchemT, ggchemp, ggchemz, spec, ggchemabn = cheminfo
        tic = time.time()
        nspec, nump, numt, numz = ggchemabn.shape
        abn = np.zeros((nspec, nlayer, ncolumn))
        
        if not np.all(np.isclose(p, ggchemp)):
            print("Pressures of fit and chemistry do not match. Exiting.")
            sys.exit()

        # TauREx wants H and e- for H- opacity, so we need to calculate
        # those abundances. This avoids requiring users to include H and e-
        # in their requested composition, which would be unintuitive
        if 'H-' in mols:
            exmols = ['H', 'e-']
        else:
            exmols = []

        for s in range(nspec):
            if spec[s] in mols or spec[s] in exmols:
                for k in range(nlayer):
                    if z in ggchemz:
                        iz = np.where(ggchemz == z)
                        fcn = spi.interp1d(ggchemT,
                                           ggchemabn[s,k,:,iz])
                        abn[s,k,ivis] = fcn(t[k,ivis])
                    else:
                        fcn = spi.interp2d(ggchemz, ggchemT,
                                           ggchemabn[s,k])
                        for i in ivis:
                            abn[s,k,ivis] = fcn(z, t[k,ivis])

    else:
        print("Unrecognized atmosphere type.")
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
        

def tgrid(nlayers, ncolumn, tmaps, pmaps, pbot, ptop, params,
          nparams, modeltype, imodel, interptype='linear',
          smooth=None):
    """
    Make a 3d grid of temperatures, based on supplied temperature maps
    placed at the supplied pressures. Dimensions are (nlayers,
    ncolumns). Will optionally smooth with a rolling average.

    """
    
    temp3d = np.zeros((nlayers, ncolumn))

    logp1d = np.linspace(np.log10(pbot), np.log10(ptop), nlayers)

    if 'tbot' in modeltype:
        if 'ttop' in modeltype:
            oob = 'both'
            itop = np.where(modeltype == 'ttop')[0][0]
            ibot = np.where(modeltype == 'tbot')[0][0]
            oobparams = \
                (params[imodel[itop]],
                 params[imodel[ibot]])
        else:
            oob = 'bot'
            ibot = np.where(modeltype == 'tbot')[0][0]
            oobparams = (params[imodel[ibot]])
    else:
        if 'ttop' in modeltype3d:
            oob = 'top'
            itop = np.where(modeltype == 'ttop')[0][0]
            oobparams = (params[imodel[ibot]])
        else:
            oob = 'isothermal'

    for i in range(ncolumn):
        if oob == 'extrapolate':
            fill_value = 'extrapolate'
            p_interp = np.copy(pmaps[:,i,j])
            t_interp = np.copy(tmaps[:,i,j])
        elif oob == 'isothermal':
            imax = np.argsort(pmaps[:,i])[-1]
            imin = np.argsort(pmaps[:,i])[0]
            fill_value = (tmaps[:,i][imin], tmaps[:,i][imax])
            p_interp = np.copy(pmaps[:,i])
            t_interp = np.copy(tmaps[:,i])
        elif oob == 'top':
            ttop = oobparams[0]
            p_interp = np.concatenate((pmaps[:,i], (ptop,)))
            t_interp = np.concatenate((tmaps[:,i], (ttop,)))
            imax = np.argsort(pmaps[:,i])[-1]
            fill_value = (ttop, tmaps[:,i][imax])
        elif oob == 'bot':
            tbot = oobparams[0]
            p_interp = np.concatenate((pmaps[:,i], (pbot,)))
            t_interp = np.concatenate((tmaps[:,i], (tbot,)))
            imin = np.argsort(pmaps[:,i])[0]
            fill_value = (tmaps[:,i][imin], tbot)
        elif oob == 'both':
            ttop = oobparams[0]
            tbot = oobparams[1]
            p_interp = np.concatenate((pmaps[:,i],
                                       (ptop, pbot)))
            t_interp = np.concatenate((tmaps[:,i],
                                       (ttop, tbot)))
            fill_value = 'extrapolate' # shouldn't matter

        interp = spi.interp1d(np.log10(p_interp),
                              t_interp, kind=interptype,
                              bounds_error=False,
                              fill_value=fill_value)

        temp3d[:,i] = interp(logp1d)

        if smooth is not None:
            T = temp3d[:,i]
            Tsmooth = np.convolve(T, np.ones(smooth), 'valid') / smooth
            nedge = np.int((len(T) - len(Tsmooth)) / 2)
            temp3d[:,i][nedge:-nedge] = Tsmooth

    return temp3d, 10**logp1d

def pmaps(params, fit):
    '''
    Calculates pressures of tmaps using a variety of mapping functions.
    '''
    tmaps = fit.tmaps3d
    lat   = fit.lat3d
    lon   = fit.lon3d
    im = np.where(fit.modeltype3d == 'pmap')[0][0]
    mapfunc = fit.cfg.threed.modelnames[im]
    mapparams = params[fit.imodel3d[im]]
    
    pmaps = np.zeros(tmaps.shape)
    nmap, ncolumn = pmaps.shape
    if   mapfunc == 'isobaric':
        for i in range(nmap):
            pmaps[i] = 10.**mapparams[i]
    elif mapfunc == 'isobaric2':
        npar = 4
        for i in range(nmap):
            ip = npar * i
            lev1 = mapparams[ip]
            lev2 = mapparams[ip+1]
            lon1 = mapparams[ip+2]
            lon2 = mapparams[ip+3]
            where1 = np.where((lon <= lon1) | (lon >= lon2))
            where2 = np.where((lon >  lon1) & (lon <  lon2))
            pmaps[i][where1] = 10**lev1
            pmaps[i][where2] = 10**lev2
    elif mapfunc == 'sinusoidal':
        npar = 4
        for i in range(nmap):
            ip = npar * i
            pmaps[i] = 10.**(mapparams[ip] + \
                mapparams[ip+1]*np.cos((lat                )*np.pi/180.) + \
                mapparams[ip+2]*np.cos((lon-mapparams[ip+3])*np.pi/180.))
    # This one needs to be updated for the new 3D grid
    # elif mapfunc == 'flexible':
    #     ilat, ilon = np.where((lon + dlon / 2. > fit.minvislon) &
    #                           (lon - dlon / 2. < fit.maxvislon))
    #     nvis = len(ilat)
    #     for i in range(nmap):
    #         ip = 0
    #         for j, k in zip(ilat, ilon):
    #             pmaps[i,j,k] = 10.**mapparams[i*nvis+ip]
    #             ip += 1
    elif mapfunc == 'quadratic':
        npar = 6
        for i in range(nmap):
            ip = npar*i
            pmaps[i] = 10.**(
                mapparams[ip  ]          +
                mapparams[ip+1] * lat**2 +
                mapparams[ip+2] * lon**2 +
                mapparams[ip+3] * lat    +
                mapparams[ip+4] * lon    +
                mapparams[ip+5] * lat*lon)
    elif mapfunc == 'cubic':
        npar = 10
        for i in range(nmap):
            ip = npar*i
            pmaps[i] = 10.**(
                mapparams[ip  ]              +
                mapparams[ip+1] * lat**3     +
                mapparams[ip+2] * lon**3     +
                mapparams[ip+3] * lat**2     +
                mapparams[ip+4] * lon**2     +
                mapparams[ip+5] * lat        +
                mapparams[ip+6] * lon        +
                mapparams[ip+7] * lat**2*lon +
                mapparams[ip+8] * lat*lon**2 +
                mapparams[ip+9] * lat*lon)
    else:
        print("WARNING: Unrecognized pmap model.")

    return pmaps

def setup_GGchem(tmin, tmax, numt, pmin, pmax, nump, zmin, zmax, numz,
                 condensates=False, charges=True,
                 elements=['H', 'He', 'C', 'O', 'N'], dustfile=None,
                 dispolfiles=None):
    # Temperatures
    tgrid = np.linspace(tmin, tmax, numt)
    # Pressures 
    pgrid = np.logspace(np.log10(pmax), np.log10(pmin), nump)
    # Metallicities
    zgrid = np.linspace(zmin, zmax, numz)

    # Stuff that should probably be up to the user
    abundance_profile = 'solar'

    # Get molecules
    gg = taurex_ggchem.GGChem(metallicity=1.0,
                              selected_elements=elements,
                              abundance_profile=abundance_profile,
                              equilibrium_condensation=condensates,
                              include_charge=charges,
                              dustchem_file=dustfile,
                              dispol_files=dispolfiles)
    ng = len(gg.gases)
    if condensates:
        nc = len(gg.condensates)
        spec = np.concatenate((gg.gases, gg.condensates))
    else:
        nc = 0
        spec = gg.gases

    ns = ng + nc

    abn = np.zeros((ns, nump, numt, numz))

    pbar = progressbar.ProgressBar(max_value=nump*numt*numz)
    for iz, z in enumerate(zgrid):
        gg = taurex_ggchem.GGChem(metallicity=10**z,
                                  selected_elements=elements,
                                  abundance_profile=abundance_profile,
                                  equilibrium_condensation=condensates,
                                  include_charge=charges,
                                  dustchem_file=dustfile,
                                  dispol_files=dispolfiles)
        for it, t in enumerate(tgrid):
            for ip, p in enumerate(pgrid):
                # Convert to pascals
                gg.initialize_chemistry(nlayers=1,
                    temperature_profile=[t],
                    pressure_profile=[p * 1e5])
                abn[:ng,ip,it,iz] = gg.mixProfile.squeeze()
                if condensates:
                    abn[ng:ns,ip,it,iz] = gg.condensateMixProfile.squeeze()
                pbar.update(ip+it*nump+iz*numt*nump)

    return tgrid, pgrid, zgrid, spec, abn

    
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


def cloudmodel_to_grid(fit, p, params, abn, spec):
    '''
    Function to turn cloud models into physical properties in the
    3D grid.
    '''
    mnames = fit.cfg.threed.modelnames

    nclouds = np.sum(fit.modeltype3d == 'clouds')

    # Inelegant solution
    if 'eqclouds' in fit.cfg.threed.modelnames:
        nclouds += len(fit.cfg.threed.cmols) - 1 # already counted once

    allshape = (nclouds, fit.cfg.threed.nlayers, fit.ncolumn)
    
    allrad = np.zeros(allshape)
    allmix = np.zeros(allshape)
    allq   = np.zeros(allshape)

    ic = 0
    for i, mtype in enumerate(fit.modeltype3d):
        if mtype != 'clouds':
            continue
        
        if mnames[i] == 'leemie':
            im = np.where(fit.cfg.threed.modelnames == mnames[i])[0][0]
            leepar = params[fit.imodel3d[im]]
            
            radius  =     leepar[0]
            q0      =     leepar[1]
            mixrat  =     leepar[2]
            bottomp = 10**leepar[3]
            topp    = 10**leepar[4]

            shape = (fit.cfg.threed.nlayers,
                     fit.ncolumn)
            radii = np.zeros(shape)
            mix   = np.zeros(shape)
            q     = np.zeros(shape)

            where = np.where((p >= topp) & (p <= bottomp))
            radii[where,:] = radius
            mix[  where,:] = mixrat
            q[    where,:] = q0

            allrad[ic] = radii
            allmix[ic] = mix
            allq[ic]   = q

            ic += 1
        elif mnames[i] == 'leemie2':
            im = np.where(fit.cfg.threed.modelnames == mnames[i])[0][0]
            leepar = params[fit.imodel3d[im]]

            # Cloud 1
            radius1  =     leepar[0]
            q01      =     leepar[1]
            mixrat1  =     leepar[2]
            bottomp1 = 10**leepar[3]
            topp1    = 10**leepar[4]
            # Cloud 2
            radius2  =     leepar[5]
            q02      =     leepar[6]
            mixrat2  =     leepar[7]
            bottomp2 = 10**leepar[8]
            topp2    = 10**leepar[9]
            # Boundaries between clouds
            center   = leepar[10]
            width    = leepar[11]
            lon1     = center - width / 2.
            lon2     = center + width / 2.

            shape = (fit.cfg.threed.nlayers,
                     fit.ncolumn)
            radii = np.zeros(shape)
            mix   = np.zeros(shape)
            q     = np.zeros(shape)

            # Not efficient to do this all the time...
            p3d = p.reshape((fit.cfg.threed.nlayers, 1))
            p3d = np.tile(p3d, (1, fit.ncolumn))

            lon3d = fit.lon.reshape((1, fit.ncolumn))
            lon3d = np.tile(lon3d, (fit.cfg.threed.nlayers, 1))

            # Cloud 1
            pcond1   = (p3d >= topp1)  & (p3d <= bottomp1)
            loncond1 = (lon3d <= lon1) | (lon3d >= lon2)
            where1 = np.where(pcond1 & loncond1)

            radii[where1] = radius1
            mix[  where1] = mixrat1
            q[    where1] = q01

            # Cloud 2
            pcond2   = (p3d >= topp2)  & (p3d <= bottomp2)
            loncond2 = (lon3d >  lon1) & (lon3d <  lon2)
            where2 = np.where(pcond2 & loncond2)

            radii[where2] = radius2
            mix[  where2] = mixrat2
            q[    where2] = q02
        
            allrad[ic] = radii
            allmix[ic] = mix
            allq[ic]   = q

            ic += 1
        elif mnames[i] == 'leemie-clearspot':
            im = np.where(fit.cfg.threed.modelnames == mnames[i])[0][0]
            leepar = params[fit.imodel3d[im]]
            
            radius  =     leepar[0]
            q0      =     leepar[1]
            mixrat  =     leepar[2]
            bottomp = 10**leepar[3]
            topp    = 10**leepar[4]
            center  =     leepar[5]
            width   =     leepar[6]
            
            lon1    =  center - width / 2.
            lon2    =  center + width / 2.

            shape = (fit.cfg.threed.nlayers,
                     fit.cfg.twod.nlat,
                     fit.cfg.twod.nlon)
            radii = np.zeros(shape)
            mix   = np.zeros(shape)
            q     = np.zeros(shape)

            # Not efficient to do this all the time...
            p3d = p.reshape((fit.cfg.threed.nlayers, 1))
            p3d = np.tile(p3d, (1, fit.ncolumn))

            lon3d = fit.lon.reshape((1, fit.ncolumn))
            lon3d = np.tile(lon3d, (fit.cfg.threed.nlayers, 1))

            pcond   = (p3d >= topp)   & (p3d <= bottomp)
            loncond = (lon3d <= lon1) | (lon3d >= lon2)
            where = np.where(pcond & loncond)

            radii[where] = radius
            mix[  where] = mixrat
            q[    where] = q0

            allrad[ic] = radii
            allmix[ic] = mix
            allq[ic]   = q

            ic += 1
        elif mnames[i] == 'eqclouds':
            im = np.where(fit.cfg.threed.modelnames == mnames[i])[0][0]
            par = params[fit.imodel3d[im]]
            
            radius = par[0]
            q0     = par[1]
            
            shape = (fit.cfg.threed.nlayers,
                     fit.ncolumn)
            for s in range(len(spec)):
                if spec[s] in fit.cfg.threed.cmols:
                    radii = np.zeros(shape)
                    q     = np.zeros(shape)

                    where = np.where(abn[s] != 0)
                    
                    radii[where] = radius
                    q[    where] = q0

                    allmix[ic] = abn[s]
                    allrad[ic] = radii
                    allq[ic]   = q

                    ic += 1
                
        else:
            print("Cloud model {} not recognized.".format(mnames[i]))

    return allrad, allmix, allq
    
        
        
        
    
    
        
