Scripts
=======

This section contains some scripts that may be useful for your applications
of the ThERESA code. They are not needed to run ThERESA.

synthlc
-------

This script generates synthetic light curves, given a 3D description
of an exoplanet atmosphere temperature structure (a GCM) and
parameters for the planet, star, radiative transfer, and observation.
Many of the parameters are similar to those used by ThERSA. Note that
in order to run this code you need a 3D temperature structure in the
right format and a set of opacities/CIAs for Tau-REx, which can be
fetched from a number of locations (e.g., ExoTransmit, HITRAN/HITEMP).
The tutorial (:doc:`gettingstarted`) includes a script to fetch some
of the ExoTransmit opacities.

To run synthlc, do the following:

.. code-block:: bash

   cd scripts
   ./synthlc.py synthlc.cfg

If the configuration file is set up correctly, this will read the GCM
temperature structure, calculate thermochemical equilibrium, run
radiative transfer, and integrate over the planet at the observation
times requested. It will produce time.txt, flux.txt, and ferr.txt,
which can be used in ThERESA.

Options
^^^^^^^

* planetname -- Name of the simulated planet.
* outdir -- Where to store the output. Will be created if it does not exist.
* ms -- Stellar mass in solar masses.
* rs -- Stellar radius in solar radii.
* ts -- Stellar temperature [K].
* ds -- Distance to star [pc]. Has no effect at the moment.
* zs -- Stellar metallicity [dex].
* mp -- Planetary mass in Jupiter masses.
* rp -- Planetary radius in Jupiter radii.
* ap -- Planet semimajor axis [au].
* bp -- Planet impact parameter.
* porb -- Planet orbital period [days]
* prot -- Planet rotational period [days]
* t0 -- Planet time of transit [days]
* ecc -- Planet eccentricity.
* inc -- Planet inclination [deg].
* atmtype -- Same as ThERESA. See :doc:`configuration`.
* atmfile -- Same as ThERESA. See :doc:`configuration`.
* mols -- Same as ThERESA. See :doc:`configuration`.
* opacdir -- Location of TauREx molecular opacities.
* ciadir  -- Location of TauREx CIA opacities.
* filtdir -- Location of filter files.
* filters -- List of filter file names in filtdir.
* phasestart -- When to start the observation in orbital phase.
* phaseend -- When to end the observation in orbital phase.
* dt -- Length of one exposure or integration of the observation [s].
* noise -- List of star-normalized noise estimates for each filter.
* necl -- Number of stacked eclipses. Uncertainties scale as :math:`\sqrt{\textrm{necl}}`.
* oom -- Number of orders of magnitude of pressure change in the GCM.
* surfp -- Surface pressure of the GCM.
* gcmfile -- File describing GCM output.
