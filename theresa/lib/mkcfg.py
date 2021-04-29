import os
import numpy as np
import configparser as cp

libdir = os.path.dirname(os.path.realpath(__file__))

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

    if "molfile" not in keys:
        f.write("{:s} {:s}\n".format('molfile', os.path.join(libdir,
                                                             "modules",
                                                             "transit",
                                                             "inputs",
                                                             "molecules.dat")))

    f.close()

    return 'transit.cfg'

        
