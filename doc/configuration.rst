Configuration
=============

Users supply inputs and setting to ThERESA through a configuration
file, following the ConfigParser format. The file is split into
several sections:

* General
* 2D
* 3D
* Star
* Planet
* taurex

General Settings
----------------

This section has one option -- the location of output (for 2D mode)
or input and output (for 3D mode).

* outdir -- The directory where output will be written. The directory
  is created if it does not exist.

2D Settings
-----------

This section contains options for the 2D fits.

* ncpu -- Number of CPUs to use in parallel. Also sets the number of
  chains in the MCMC.
* nsamples -- Number of total iterations in the MCMC.
* burnin -- Number of \'\'burned-in\'\' (discarded) iterations per
  Markov chain.
* lmax -- Largest :math:`l` value to use when generating the spherical
  harmonic basis maps.
* ncurves -- Number of eigencurves used in the fits.
* pca -- Type of principle compoenent analysis to use. Options are \'pca\'
  for typical PCA and \'tsvd\' for truncated singular-value decomposition,
  which does not do mean subtraction to ensure physically plausible
  eigencurves. \'tsvd\' is the recommended option.
* ncalc -- Number of calculations for post-MCMC analysis. If higher than
  nsamples, it will be reduced to equal nsamples.
* leastsq -- Controls whether to do a least-squares minimization prior
  to fitting, and if so, which kind. Options are \'None\', \'trf\'
  (respect parameter boundaries), and \'lm\' (faster, does not respect
  boundaries).
* nlat, nlon -- Number of evenly spaced latitude and longitude bins
  for plotting, visibility function calculation, and, later, 3D
  modeling.
* posflux -- Boolean to enforce positive flux in visible grid cells.
  Setting this to False may prevent future 3D modeling.
* timefile -- Path to a file that lists the times of observation.
* fluxfile -- Path to a file that lists the fluxes of the observation.
* ferrfile -- Path to a file that lists the flux uncertainties of the
  observation.
* filtfiles -- List of files that contain transmission information for
  the filters that correspond to the fluxes of the observation.
* plots -- Boolean to turn plotting on or off.
* animations -- Boolean to turn animations on or off.


3D Settings
-----------

This section contains options for the 3D fit.

* ncpu -- Number of CPUs to use in parallel. Also sets the number of
  chains in the MCMC.
* nsamples -- Number of total iterations in the MCMC.
* burnin -- Number of \'\'burned-in\'\' (discarded) iterations per
  Markov chain.
* leastsq -- Controls whether to do a least-squares minimization prior
  to fitting, and if so, which kind. Options are \'None\', \'trf\'
  (respect parameter boundaries), and \'lm\' (faster, does not respect
  boundaries). The complexity of the 3D model usually makes a
  least-squares optimization difficult.
* elemfile -- Path to a file that describes elemental ratios.
* atmtype -- Type of atomspheric composition to use. Options are
  \'rate\' and \'GGchem\'. See :doc:`codedesc` for more information.
* atmfile -- If using a GGchem composition, this is a path to the
  GGchem output file.
* nlayers -- Number of layers in the fitted atmosphere, equally spaced in
  log-pressure.
* ptop -- Pressure at the top of the atmosphere, in bars.
* pbot -- Pressure at the bottom of the atmosphere, in bars.
* rtfunc -- Radiative transfer function to use. Currently limited to
  \'taurex\'.
* mapfunc -- Function to link 2D maps to pressure levels in the 3D
  model.  Options are \'isobaric\', \'sinusoidal\', and
  \'flexible\'. See :doc:`codedesc` for more information.
* oob -- Out-of-bounds behavior when extrapolating the 3D thermal
  structure. Options are \'isothermal\', \'top\' (add a parameter for
  top-of-the-atmosphere temperature), \'bot\' (add a parameter for
  bottom-of-the-atmosphere temperature), and \'both\'.
* interp -- Method to use when interpolating the 3D thermal structure.
  Options are \'linear\', \'quadratic\', and \'cubic\'.
* fitcf -- Boolean to turn on or off enforced contribution function
  consistency.
* mols -- List of molecules which will have opacity in the radiative
  transfer calculation.
* params (optional) -- Sets the starting values for the parameters in the MCMC.
* pmin (optional) -- Sets the lower boundaries for the parameters in the MCMC.
* pmax (optional) -- Sets the upper boundaries for the parameters in the MCMC.
* pnames (optional) -- Sets the names of the MCMC parameters, for plotting.
* plots -- Boolean to turn plotting on or off.
* animations -- Boolean to turn animations on or off.


Star
----

Stellar parameters.

* m -- Mass in solar masses.
* r -- Radius in solar radii.
* prot -- Rotational period in days.
* t -- Temperature in K.
* d -- Distance in parsecs.
* z -- Metallicity (dex relative to solar).


Planet
------

Planetary parameters.

* m -- Mass in solar masses.
* r -- Radius in solar radii.
* p0 -- Pressure at r (bars).
* porb -- Orbital period in days.
* prot -- Rotational period in days.
* Omega -- Longitude of ascending node in degrees.
* ecc -- Eccentricity.
* inc -- Inclination in degrees. 90 is considered edge-on.
* b -- Impact parameter.
* w -- Longitude of periastron in degrees.
* a -- Semi-major axis in AU.
* t0 -- Time of transit in days.


taurex
------

Tau-REx specific options.

* csxdir -- Directory containing molecular opacity data. This directory
  must contain a file or files for each molecule in the 3D fit.
* ciadir -- Directory containing collision-induced absorption cross
  section files.
* wnlow -- Minimum wavenumber to calculate radiative transfer.
* wnhigh -- Maximum wavenumber to calculate radiative transfer.
  
