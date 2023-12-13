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

        nobs = 0
        for section in config.sections():
            if section.startswith('Observation'):
                nobs += 1

        ninst = 0
        for section in config.sections():
            if section.startswith('Instrument'):
                ninst += 1

        self.cfg = cc.Configuration(nobs, ninst)

        self.cfg.cfile = cfile
        self.cfg.cfg   = config
       
        # 2D options
        self.cfg.twod.outdir = self.cfg.cfg.get('2D', 'outdir')
        
        self.cfg.twod.lmax    = self.cfg.cfg.getint('2D', 'lmax')
        self.cfg.twod.ncurves = self.cfg.cfg.getint('2D', 'ncurves')
            
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

        if self.cfg.cfg.has_option('2D', 'orbcheck'):
            self.cfg.twod.orbcheck = self.cfg.cfg.get('2D', 'orbcheck')
            if self.cfg.twod.orbcheck == 't0':
                self.cfg.twod.sigorb = [
                    float(a) for a in self.cfg.cfg.get('2D', 'sigorb').split()]
        else:
            self.cfg.twod.orbcheck = None
            self.cfg.twod.sigorb   = None
            
        # 3D options
        self.cfg.threed.outdir = self.cfg.cfg.get('3D', 'outdir')
        self.cfg.threed.indir  = self.cfg.cfg.get('3D', 'indir')
        
        self.cfg.threed.ncpu       = self.cfg.cfg.getint('3D', 'ncpu')
        self.cfg.threed.nsamples   = self.cfg.cfg.getint('3D', 'nsamples')
        self.cfg.threed.burnin     = self.cfg.cfg.getint('3D', 'burnin')
        
        self.cfg.threed.ptop    = self.cfg.cfg.getfloat('3D', 'ptop')
        self.cfg.threed.pbot    = self.cfg.cfg.getfloat('3D', 'pbot')
        self.cfg.threed.atmtype = self.cfg.cfg.get(     '3D', 'atmtype')
        self.cfg.threed.nlayers = self.cfg.cfg.getint(  '3D', 'nlayers')
        
        self.cfg.threed.rtfunc  = self.cfg.cfg.get('3D', 'rtfunc')

        self.cfg.threed.modelnames = np.array(
            self.cfg.cfg.get('3D', 'models').split())

        self.cfg.threed.interp  = self.cfg.cfg.get('3D', 'interp')

        if 'z' not in self.cfg.threed.modelnames:
            try:
                self.cfg.threed.z = self.cfg.cfg.getfloat('3D', 'z')
            except:
                print("Must specify metallicity if not fitting to it.")
                sys.exit()
        else:
            self.cfg.threed.z = 'fit'

        self.cfg.threed.elem = self.cfg.cfg.get('3D', 'elem').split()

        self.cfg.threed.mols  = self.cfg.cfg.get('3D', 'mols').split()
        if 'eqclouds' in self.cfg.threed.modelnames:
            self.cfg.threed.cmols = self.cfg.cfg.get('3D', 'cmols').split()
        else:
            self.cfg.threed.cmols = []

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

        if self.cfg.threed.atmtype == 'ggchem':
            self.cfg.threed.tmin = self.cfg.cfg.getfloat('3D', 'tmin')
            self.cfg.threed.tmax = self.cfg.cfg.getfloat('3D', 'tmax')
            self.cfg.threed.numt = self.cfg.cfg.getint(  '3D', 'numt')
            self.cfg.threed.zmin = self.cfg.cfg.getfloat('3D', 'zmin')
            self.cfg.threed.zmax = self.cfg.cfg.getfloat('3D', 'zmax')
            self.cfg.threed.numz = self.cfg.cfg.getint(  '3D', 'numz')
            self.cfg.threed.condensates = \
                self.cfg.cfg.getboolean('3D', 'condensates')

        self.cfg.threed.taulimit = self.cfg.cfg.getfloat('3D', 'taulimit')
            
        # Star options
        self.cfg.star.m    = self.cfg.cfg.getfloat('Star', 'm')
        self.cfg.star.r    = self.cfg.cfg.getfloat('Star', 'r')
        self.cfg.star.prot = self.cfg.cfg.getfloat('Star', 'prot')
        self.cfg.star.t    = self.cfg.cfg.getfloat('Star', 't')
        self.cfg.star.d    = self.cfg.cfg.getfloat('Star', 'd')
        self.cfg.star.z    = self.cfg.cfg.getfloat('Star', 'z')

        if self.cfg.cfg.has_option('Star', 'starspec'):
            self.cfg.star.starspec = self.cfg.cfg.get('Star', 'starspec')
        else:
            print('Using default blackbody spectrum for star.')
            self.cfg.star.starspec = 'bbint'
            
        if self.cfg.star.starspec == 'custom':
            if self.cfg.cfg.has_option('Star', 'starspecfile'):
                self.cfg.star.starspecfile = \
                    self.cfg.cfg.get('Star', 'starspecfile')
            else:
                print("Must specify stellar spectrum file "
                      "using starspecfile.")

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

        # Instruments
        for i, inst in enumerate(self.cfg.instruments):
            section = "Instrument{}".format(i+1)
            inst.name = self.cfg.cfg.get(section, 'name')
            inst.filtfiles = self.cfg.cfg.get(section, 'filtfiles').split()

        # Observations
        for i, obs in enumerate(self.cfg.observations):
            section = "Observation{}".format(i+1)
            obs.timefile = self.cfg.cfg.get(section, 'timefile')
            obs.fluxfile = self.cfg.cfg.get(section, 'fluxfile')
            obs.ferrfile = self.cfg.cfg.get(section, 'ferrfile')

            obs.name = self.cfg.cfg.get(section, 'name')
            obs.instrument = self.cfg.cfg.getint(section, 'instrument')

            if self.cfg.cfg.has_option(section, 'baseline'):
                obs.baseline = self.cfg.cfg.get(section, 'baseline')
                if (obs.baseline == 'None') or \
                   (obs.baseline == 'none'):
                    obs.baseline = None
            else:
                obs.baseline = None

            if self.cfg.cfg.has_option(section, 'clip'):
                obs.clip = np.array(
                    [float(a) for a in self.cfg.cfg.get(section,
                                                        'clip').split()])
                if len(obs.clip) % 2 != 0:
                    msg = "Uneven number of clips for observation {}."
                    print(msg.format(obs.name))
                    sys.exit()
            else:
                obs.clip = None                             
        
    def read_data(self):
        '''
        Read data files, including a stellar spectrum if provided.
        Populate related attributes.
        '''
        # The Data objects link Observations to Instruments. Every
        # Instrument has an associated Data object, and within each
        # Data object there is a Visit object for each Observation
        # object attached to that instrument. ThERESA will generate a
        # map (and Map object) for each Data object. Data and Visit
        # objects contain processed products, while Instrument and
        # Observation objects only contain parsed configuration
        # options.
        self.datasets = []
        for ii, inst in enumerate(self.cfg.instruments):
            data = Data()
            data.visits = []
            data.filtfiles = inst.filtfiles
            data.name = inst.name

            for obs in self.cfg.observations:
                if obs.instrument == ii + 1:
                    visit = Visit()
                         
                    # Unclipped
                    visit.tuc    = np.loadtxt(obs.timefile, ndmin=1)
                    visit.fluxuc = np.loadtxt(obs.fluxfile, ndmin=2).T
                    visit.ferruc = np.loadtxt(obs.ferrfile, ndmin=2).T

                    visit.timefile  = obs.timefile
                    visit.fluxfile  = obs.fluxfile
                    visit.ferrfile  = obs.ferrfile

                    visit.name       = obs.name
                    visit.instrument = obs.instrument
                    visit.baseline   = obs.baseline

                    visit.clip = obs.clip

                    if visit.clip is None:
                        visit.t    = np.copy(visit.tuc)
                        visit.flux = np.copy(visit.fluxuc)
                        visit.ferr = np.copy(visit.ferruc)
                    else:
                        nclip = len(obs.clip) // 2
                        whereclip = np.ones(len(visit.tuc), dtype=bool)
                        for i in range(nclip):
                            whereclip[(visit.tuc > visit.clip[2*i  ]) &
                                      (visit.tuc < visit.clip[2*i+1])] = False
                        visit.t    = np.copy(visit.tuc[whereclip])
                        visit.flux = np.copy(visit.fluxuc[:,whereclip])
                        visit.ferr = np.copy(visit.ferruc[:,whereclip])

                    visit.tloc = visit.t - np.min(visit.t)

                    if len(visit.t) != visit.flux.shape[1]:
                        print("WARNING: Number of times does not match" +
                              "the size of the flux array.")
                        sys.exit()

                    if len(visit.t) != visit.ferr.shape[1]:
                        print("WARNING: Number of times does not match" +
                              "the size of the ferr array.")
                        sys.exit()

                    data.visits.append(visit)

            # Concatenated data arrays for convenience
            data.t = np.concatenate([v.t for v in data.visits])
            data.flux = np.concatenate([v.flux for v in data.visits], axis=1)
            data.ferr = np.concatenate([v.ferr for v in data.visits], axis=1)
            
            self.datasets.append(data)

        if hasattr(self.cfg.star, 'starspecfile'):
            self.starwl, self.starflux = np.loadtxt(
                self.cfg.star.starspecfile, unpack=True)
        else:
            self.starwl, self.starflux = None, None

    def read_filters(self):
        '''
        Read filter files and populate attributes.
        '''
        for data in self.datasets:
            filtwl, filtwn, filttrans, wnmid, wlmid = \
                utils.readfilters(data.filtfiles)
            data.filtwl    = filtwl
            data.filtwn    = filtwn
            data.filttrans = filttrans
            data.wnmid     = wnmid
            data.wlmid     = wlmid           
            
    def save(self, outdir, fname=None):
        '''
        Save a Fit object to a pickle file.

        Arguments
        ---------
        outdir: String
            Directory where the file will be saved.

        fname: String
            Optional name of the file. Default is fit.pkl

        '''
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

class LN:
    '''
    A class to hold result from a fit with a single combination of 
    lmax and ncurves.
    '''
    pass

class Data:
    '''
    A class to hold information about a single Instrument. Can be
    spectroscopic, with multiple filters, flux arrays, and uncertainty
    arrays.

    '''
    pass

class Visit:
    '''
    A class to hold information about a single Observation (or visit)
    with an instrument. Stored as an attribute to a Data object.
    '''

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
