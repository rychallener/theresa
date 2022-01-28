# ThERESA

A code for retrieving three-dimensional maps of exoplanets.

The code constructs 2-dimensional maps for each light given light
curve, places those maps vertically in an atmosphere, runs radiative
transfer to calculate emission from the planet over a latitude/longitude
grid, and integrates over the grid (combined with the visibility
function) to generate light curves. These light curves are compared
against the input light curves behind MCMC to explore parameter space.

## Developers

Ryan C. Challener (rchallen@umich.edu)

Emily Rauscher

Lucas Brefka

## Citation

If you use this code in your work, please cite our paper. Thanks!

https://ui.adsabs.harvard.edu/abs/2021arXiv211008159C/abstract