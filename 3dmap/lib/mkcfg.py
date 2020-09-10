import os
import numpy as np
import configparser as cp

def mktransit(cfile, outdir):
    """
    Parse transit configuration from ConfigParser format in the 
    main configuration file to the format transit desires.
    """
    maincfg = cp.ConfigParser()
    maincfg.read([cfile])

    tfile = os.path.join(outdir, 'transit.cfg')
    f = open(tfile, 'w')

    keys = maincfg.options('transit')

    for key in keys:
        val = maincfg.get('transit', key) 
        f.write("{:s} {:s}\n".format(key, val))

    return tfile

        
