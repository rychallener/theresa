import os
import sys
import numpy as np
import pickle
import configparser as cp
import configclass as cc
import scipy.constants as sc

import utils

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
        self.cfg.outdir     = self.cfg.cfg.get('General', 'outdir')
       
        # 2D options
        self.cfg.twod.timefile = self.cfg.cfg.get('2D', 'timefile')
        self.cfg.twod.fluxfile = self.cfg.cfg.get('2D', 'fluxfile')
        self.cfg.twod.ferrfile = self.cfg.cfg.get('2D', 'ferrfile')

        self.cfg.twod.filtfiles = self.cfg.cfg.get('2D', 'filtfiles').split()
        nfilt = len(self.cfg.twod.filtfiles)
        
        if len(self.cfg.cfg.get('2D', 'lmax').split()) == 1:
            self.cfg.twod.lmax = np.ones(nfilt, dtype=int) * \
                self.cfg.cfg.getint('2D', 'lmax')
        else:
            self.cfg.twod.lmax = np.array(
                [int(a) for a in self.cfg.cfg.get('2D', 'lmax').split()])

        if len(self.cfg.cfg.get('2D', 'ncurves').split()) == 1:
            self.cfg.twod.ncurves = np.ones(nfilt, dtype=int) * \
                self.cfg.cfg.getint('2D', 'ncurves')
        else:
            self.cfg.twod.ncurves = np.array(
                [int(a) for a in self.cfg.cfg.get('2D', 'ncurves').split()])
            
        self.cfg.twod.pca        = self.cfg.cfg.get(   '2D', 'pca')
        self.cfg.twod.ncalc      = self.cfg.cfg.getint('2D', 'ncalc')
        self.cfg.twod.ncpu       = self.cfg.cfg.getint('2D', 'ncpu')
        self.cfg.twod.nsamples   = self.cfg.cfg.getint('2D', 'nsamples')
        self.cfg.twod.burnin     = self.cfg.cfg.getint('2D', 'burnin')
              
        self.cfg.twod.posflux = self.cfg.cfg.getboolean('2D', 'posflux')
        
        self.cfg.twod.nlat    = self.cfg.cfg.getint('2D', 'nlat')
        self.cfg.twod.nlon    = self.cfg.cfg.getint('2D', 'nlon')

        self.cfg.twod.plots      = self.cfg.cfg.getboolean('2D', 'plots')
        self.cfg.twod.animations = self.cfg.cfg.getboolean('2D', 'animations')
        
        self.cfg.twod.leastsq = self.cfg.cfg.get('2D', 'leastsq')
        if (self.cfg.twod.leastsq == 'None' or
            self.cfg.twod.leastsq == 'False'):
            self.cfg.twod.leastsq = None

        if self.cfg.cfg.has_option('2D', 'fgamma'):
            self.cfg.twod.fgamma = self.cfg.cfg.getfloat('2D', 'fgamma')
        else:
            self.cfg.twod.fgamma = 1.0

        # 3D options
        self.cfg.threed.ncpu       = self.cfg.cfg.getint('3D', 'ncpu')
        self.cfg.threed.nsamples   = self.cfg.cfg.getint('3D', 'nsamples')
        self.cfg.threed.burnin     = self.cfg.cfg.getint('3D', 'burnin')
        
        self.cfg.threed.elemfile = self.cfg.cfg.get('3D', 'elemfile')
        
        self.cfg.threed.ptop    = self.cfg.cfg.getfloat('3D', 'ptop')
        self.cfg.threed.pbot    = self.cfg.cfg.getfloat('3D', 'pbot')
        self.cfg.threed.atmtype = self.cfg.cfg.get(     '3D', 'atmtype')
        self.cfg.threed.atmfile = self.cfg.cfg.get(     '3D', 'atmfile')
        self.cfg.threed.nlayers = self.cfg.cfg.getint(  '3D', 'nlayers')
        
        self.cfg.threed.rtfunc  = self.cfg.cfg.get('3D', 'rtfunc')
        self.cfg.threed.mapfunc = self.cfg.cfg.get('3D', 'mapfunc')
        self.cfg.threed.oob     = self.cfg.cfg.get('3D', 'oob')
        self.cfg.threed.interp  = self.cfg.cfg.get('3D', 'interp')

        self.cfg.threed.mols = self.cfg.cfg.get('3D', 'mols').split()

        self.cfg.threed.plots      = self.cfg.cfg.getboolean('3D', 'plots')
        self.cfg.threed.animations = self.cfg.cfg.getboolean('3D', 'animations')

        self.cfg.threed.leastsq = self.cfg.cfg.get('3D', 'leastsq')
        if (self.cfg.threed.leastsq == 'None' or
            self.cfg.threed.leastsq == 'False'):
            self.cfg.threed.leastsq = None

        if self.cfg.cfg.has_option('3D', 'grbreak'):
            self.cfg.threed.grbreak = self.cfg.cfg.getfloat('3D', 'grbreak')
        else:
            self.cfg.threed.grbreak = 0.0
        
        self.cfg.threed.smooth  = self.cfg.cfg.get('3D', 'smooth')
        if self.cfg.threed.smooth == 'None':
            self.cfg.threed.smooth = None
        else:
            self.cfg.threed.smooth = np.int(self.cfg.threed.smooth)

        self.cfg.threed.fitcf = self.cfg.cfg.getboolean('3D', 'fitcf')

        for item in ['params', 'pmin', 'pmax', 'pstep']:
            if self.cfg.cfg.has_option('3D', item):
                value = np.array(
                    self.cfg.cfg.get('3D', item).split()).astype(float)
                setattr(self.cfg.threed, item, value)

        if self.cfg.cfg.has_option('3D', 'pnames'):
            self.cfg.threed.pnames = \
                self.cfg.cfg.get('3D', 'pnames').split()

        self.cfg.threed.resume = self.cfg.cfg.getboolean('3D', 'resume')

        if self.cfg.cfg.has_option('3D', 'fgamma'):
            self.cfg.threed.fgamma = self.cfg.cfg.getfloat('3D', 'fgamma')
        else:
            self.cfg.threed.fgamma = 1.0
       
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
        self.t    = np.loadtxt(self.cfg.twod.timefile, ndmin=1)
        self.flux = np.loadtxt(self.cfg.twod.fluxfile, ndmin=2).T
        self.ferr = np.loadtxt(self.cfg.twod.ferrfile, ndmin=2).T

        if len(self.t) != self.flux.shape[1]:
            print("WARNING: Number of times does not match the size " +
                  "of the flux array.")
            sys.exit()

        if len(self.t) != self.ferr.shape[1]:
            print("WARNING: Number of times does not match the size " +
                  "of the ferr array.")
            sys.exit()

    def read_filters(self):
        self.filtwl, self.filtwn, self.filttrans, self.wnmid, self.wlmid = \
            utils.readfilters(self.cfg.twod.filtfiles)           
            
    def save(self, outdir, fname=None):
        # Note: starry objects are not pickleable, so they
        # cannot be added to the Fit object as attributes. Possible
        # workaround by creating a custom Pickler?
        if type(fname) == type(None):
            fname = 'fit.pkl'

        with open(os.path.join(outdir, fname), 'wb') as f:
            pickle.dump(self, f)

class Map:
    '''
    A class to hold results from a fit to a single wavelength (a 2d map).
    '''
    pass

def load(outdir=None, filename=None):
    """
    Load a Fit object from file.
    
    Arguments
    ---------
    outdir: string
        Location of file to load. Default is an empty string (current
        directory)

    filename: string
        Name of the file to load. Default is 'fit.pkl'.

    Returns
    -------
    fit: Fit instance
        Fit object loaded from filename
    """
    if type(outdir) == type(None):
        outdir = ''
        
    if type(filename) == type(None):
        filename = 'fit.pkl'
        
    with open(os.path.join(outdir, filename), 'rb') as f:
        return pickle.load(f)
