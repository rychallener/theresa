import numpy as np

# Some useful constants

# http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
# Retrieved September 10, 2020
Mjup = 1.89819e27 # kg
Rjup = 7.1492e7   # m (equatorial)

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
# Retrieved September 11, 2020
Msun = 1.9885e30 # kg
Rsun = 6.957e8   # m (volumetric mean)

# Unit conversions
deg2rad = np.pi / 180.
rad2deg = 180. / np.pi
