import configparser as cp

class Configuration:
    """
    A class to hold parameters from a configuration file.
    """
    pass

def read_config(cfile):
    """
    Read a configuration file and return a filled-in Configuration object
    """
    config = cp.ConfigParser()
    config.read(cfile)
    cfg = Configuration()

    cfg.cfile = cfile
    cfg.cfg   = config

    cfg.lmax     = cfg.cfg.getint(    'General', 'lmax')
    cfg.outdir   = cfg.cfg.get(       'General', 'outdir')
    cfg.mkplots  = cfg.cfg.getboolean('General', 'mkplots')
    cfg.ncurves  = cfg.cfg.getint(    'General', 'ncurves')
    cfg.datafile = cfg.cfg.get(       'General', 'datafile')

    return cfg    
