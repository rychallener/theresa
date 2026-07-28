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

# Every 0.1 dex
numco = 21
comin = -2.0
comax = 0.0

dispolfiles = None

condensates = False

elem = ['H', 'He', 'C', 'N', 'O', 'S']

mols = ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'C2H2', 'C2H4', 'HCN', 'H2S']

cmols = []

cheminfo = atm.setup_GGchem(tmin, tmax, numt,
                            ptop, pbot, nlayers,
                            zmin, zmax, numz,
                            comin, comax, numco,
                            mols, cmols,
                            condensates=condensates,
                            elements=elem,
                            dispolfiles=dispolfiles)

np.savez('ggchem-default.npz', T=cheminfo[0], P=cheminfo[1], z=cheminfo[2],
         co=cheminfo[3], spec=cheminfo[4], abn=cheminfo[5])

                            
