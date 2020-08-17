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

    return cfg    
