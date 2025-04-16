#! /usr/bin/env python

import sys
sys.path.append('../lib')
import atm

import numpy as np

import taurex.log
taurex.log.disableLogging()

# Default settings
nlayers = 100
ptop = 1e-6
pbot = 1e2

# Every 50 K
numt = 77
tmin = 150
tmax = 4000

# Every 0.1 dex
numz = 41
zmin = -2.0
zmax =  2.0

dispolfiles = None

condensates = False

elem = ['H', 'He', 'C', 'N', 'O', 'S']

cheminfo = atm.setup_GGchem(tmin, tmax, numt,
                            ptop, pbot, nlayers,
                            zmin, zmax, numz,
                            condensates=condensates,
                            elements=elem,
                            dispolfiles=dispolfiles)

np.savez('ggchem-default.npz', T=cheminfo[0], P=cheminfo[1], z=cheminfo[2],
         spec=cheminfo[3], abn=cheminfo[4])

                            
