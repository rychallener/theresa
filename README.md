# ThERESA

A code for retrieving three-dimensional maps of exoplanets.

ThERESA does both 2D (latitude and longitude) and 3D (latitude,
longitude, and pressure) mapping of exoplanet atmospheres. 2D mapping
is done using the "eigenmapping" method ([Rauscher et al.,
2018](https://ui.adsabs.harvard.edu/abs/2018AJ....156..235R/abstract)).
3D mapping uses a 3D temperature and composition parameterization to
construct atmospheres, runs radiative transfer to calculate emission
from the planet over a latitude/longitude grid, and integrates over
the grid to generate light curves. These light curves are compared
against the input light curves behind MCMC to explore parameter space.

See the [documentation](https://theresa.readthedocs.io/en/latest/) for
more information.

## Developers

Ryan C. Challener (rcc276@cornell.edu)

Emily Rauscher

Lucas Brefka

Abrar Amin

## Citation

If you use this code in your work, please cite our paper. Thanks!

https://ui.adsabs.harvard.edu/abs/2022AJ....163..117C/abstract