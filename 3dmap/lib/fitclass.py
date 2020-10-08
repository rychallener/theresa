import os
import sys
import numpy as np
import pickle
import configparser as cp
import configclass as cc
import scipy.constants as sc

class Fit:
    """
    A class to hold attributes and methods related to fitting a model
    or set of models to data.
    """
    def read_config(self, cfile):
        """
        Read a configuration file and set up attributes accordingly.

        Note that self.cfg is a Configuration instance, and self.cfg.cfg
        is a raw ConfigParser instance. The ConfigParser instance should
        be parsed into attributes of the Configuration() instance for
        simpler access within other routines that use the Fit class.
        """
        config = cp.ConfigParser()
        config.read(cfile)
        self.cfg = cc.Configuration()

        self.cfg.cfile = cfile
        self.cfg.cfg   = config

        # General options
        self.cfg.lmax     = self.cfg.cfg.getint(    'General', 'lmax')
        self.cfg.outdir   = self.cfg.cfg.get(       'General', 'outdir')
        self.cfg.mkplots  = self.cfg.cfg.getboolean('General', 'mkplots')
        self.cfg.ncurves  = self.cfg.cfg.getint(    'General', 'ncurves')
        self.cfg.ncpu     = self.cfg.cfg.getint(    'General', 'ncpu')
        self.cfg.nsamples = self.cfg.cfg.getint(    'General', 'nsamples')
        self.cfg.burnin   = self.cfg.cfg.getint(    'General', 'burnin')
        self.cfg.leastsq  = self.cfg.cfg.get(       'General', 'leastsq')

        self.cfg.timefile = self.cfg.cfg.get('General', 'timefile')
        self.cfg.fluxfile = self.cfg.cfg.get('General', 'fluxfile')
        self.cfg.ferrfile = self.cfg.cfg.get('General', 'ferrfile')
        self.cfg.wlfile   = self.cfg.cfg.get('General', 'wlfile')

        self.cfg.filtfiles = self.cfg.cfg.get('General', 'filtfiles').split()

        self.cfg.atmtype = self.cfg.cfg.get(     'General', 'atmtype')
        self.cfg.atmfile = self.cfg.cfg.get(     'General', 'atmfile')
        self.cfg.nlayers = self.cfg.cfg.getint(  'General', 'nlayers')
        self.cfg.res     = self.cfg.cfg.getint(  'General', 'res')
        self.cfg.ptop    = self.cfg.cfg.getfloat('General', 'ptop')
        self.cfg.pbot    = self.cfg.cfg.getfloat('General', 'pbot')
        self.cfg.temp    = self.cfg.cfg.getfloat('General', 'temp')
        
        if self.cfg.leastsq == 'None' or self.cfg.leastsq == 'False':
            self.cfg.leastsq = None

        self.cfg.rtfunc  = self.cfg.cfg.get('General', 'rtfunc')
        self.cfg.mapfunc = self.cfg.cfg.get('General', 'mapfunc')

        self.cfg.elemfile = self.cfg.cfg.get('General', 'elemfile')

        self.cfg.posflux = self.cfg.cfg.getboolean('General', 'posflux')

        # Star options
        self.cfg.star.m    = self.cfg.cfg.getfloat('Star', 'm')
        self.cfg.star.r    = self.cfg.cfg.getfloat('Star', 'r')
        self.cfg.star.prot = self.cfg.cfg.getfloat('Star', 'prot')
        self.cfg.star.t    = self.cfg.cfg.getfloat('Star', 't')
        self.cfg.star.d    = self.cfg.cfg.getfloat('Star', 'd')
        self.cfg.star.z    = self.cfg.cfg.getfloat('Star', 'z')

        # Planet options
        self.cfg.planet.m     = self.cfg.cfg.getfloat('Planet', 'm')
        self.cfg.planet.r     = self.cfg.cfg.getfloat('Planet', 'r')
        self.cfg.planet.p0    = self.cfg.cfg.getfloat('Planet', 'p0')
        self.cfg.planet.porb  = self.cfg.cfg.getfloat('Planet', 'porb')
        self.cfg.planet.prot  = self.cfg.cfg.getfloat('Planet', 'prot')
        self.cfg.planet.Omega = self.cfg.cfg.getfloat('Planet', 'Omega')
        self.cfg.planet.ecc   = self.cfg.cfg.getfloat('Planet', 'ecc')
        self.cfg.planet.inc   = self.cfg.cfg.getfloat('Planet', 'inc')
        self.cfg.planet.w     = self.cfg.cfg.getfloat('Planet', 'w')
        self.cfg.planet.t0    = self.cfg.cfg.getfloat('Planet', 't0')
        self.cfg.planet.a     = self.cfg.cfg.getfloat('Planet', 'a')
        self.cfg.planet.b     = self.cfg.cfg.getfloat('Planet', 'b')
        
    def read_data(self):
        self.t    = np.loadtxt(self.cfg.timefile)
        self.flux = np.loadtxt(self.cfg.fluxfile).T
        self.ferr = np.loadtxt(self.cfg.ferrfile).T
        self.wl   = np.loadtxt(self.cfg.wlfile)

        if len(self.t) != self.flux.shape[1]:
            print("WARNING: Number of times does not match the size " +
                  "of the flux array.")
            sys.exit()

        if len(self.t) != self.ferr.shape[1]:
            print("WARNING: Number of times does not match the size " +
                  "of the ferr array.")
            sys.exit()

        if len(self.wl) != self.flux.shape[0]:
            print("WARNING: Number of wavelengths does not match the size " +
                  "of the flux array.")
            sys.exit()

        if len(self.wl) != self.ferr.shape[0]:
            print("WARNING: Number of wavelengths does not match the size " +
                  "of the ferr array.")
            sys.exit()

            
    def save(self, outdir, fname=None):
        # Note: starry objects are not pickleable, so they
        # cannot be added to the Fit object as attributes. Possible
        # workaround by creating a custom Pickler?
        if type(fname) == type(None):
            fname = 'fit.pkl'

        with open(os.path.join(outdir, fname), 'wb') as f:
            pickle.dump(self, f)

def load(filename):
    """
    Load a Fit object from file.
    
    Arguments
    ---------
    filename: string
        Path to the file to load

    Returns
    -------
    fit: Fit instance
        Fit object loaded from filename
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
