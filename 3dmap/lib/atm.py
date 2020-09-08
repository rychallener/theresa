import os
import sys
import numpy as np

libdir = os.path.dirname(os.path.realpath(__file__))
moddir = os.path.join(libdir, 'modules')
ratedir = os.path.join(moddir, 'rate')

sys.path.append(ratedir)
import rate

def atminit(atmtype, atmfile, nlayers, ptop, pbot, t, outdir):
    if type(t) == float or type(t) == int:
        t = np.ones(nlayers) * t

    if os.path.isfile(atmfile):
        print("Using atmosphere " + atm)
        p, t, abn = atmload(atmfile)
        atmsave(p, t, abn, outdir, atmfile)
        return p, t, abn

    p = np.logspace(pbot, ptop, nlayers)

    # Equilibrium atmosphere
    if atmtype == 'eq':
        r   = rate.Rate(C=2.5e-4, N=1.0e-4, O=5.0e-4, fHe=0.0851)
        abn = r.solve(t, p)
        atmsave(p, t, abn, outdir, atmfile)
        return p, t, abn

def atmsave(p, t, abn, outdir, atmfile):
    """
    Save an atmosphere file. Columns are pressure, temeprature, and abundance.

    Inputs:
    -------
    p: 1D array
        Pressure array

    t: 1D array
        Temperature array

    abn: 1D array
        Abundance array. Rows are abundances for each molecule, and
        columns are pressure.

    """
    nlayers = len(p)
    atmarr = np.hstack((p.reshape((nlayers, 1)),
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
    p: 1D array
        Pressure array

    t: 1D array
        Temperature array

    abn: 1D array
        Abundance array. Rows are abundances for each molecule, and
        columns are pressure.

    """
    arr = np.loadtxt(atmfile)

    p   = arr[:,0 ]
    t   = arr[:,1 ]
    abn = arr[:,2:]

    return p, t, abn
                        

    
        

    
    
        
