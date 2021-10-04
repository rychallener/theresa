Code Description
================

The goal of this package is to retrieve the three-dimensional atmosphere 
of an exoplanet from spectroscopic eclipse observations. It builds upon
the work of Rauscher et al., 2018, where they use principal component
analysis of light curves generated from spherical harmonic maps to
determine a set of orthogonal light curves (\'\'eigencurves\'\'). These
eigencurves are linearly combined to produce a best-fitting light-curve
model, which is then converted to a temperature map.

This code constructs these temperature maps for each spectroscopic
light curve and then places them vertically within the atmosphere. It
then runs a radiative transfer calculation to produce planetary
emission as a function of location on the planet. The emission is
integrated over the wavelength filters and combined with the visibility
function to create light curves to compare against the data. This
calculation is put behind a Markov-chain Monte Carlo (MCMC) to explore
parameter space.

The code is split into two operating modes: 2D and 3D. These modes are
described in detail in the following sections.

2D Mode
-------

The 2D mode of the code constructs the 2-dimensional thermal maps
which are used in the 3D mode. Hence, the code must be run in 2D
before attempting a 3D fit.

First, the code calculates light curves from positive and negative
spherical harmonics maps, up to a user-supplied complexity
(:math:`l_{\textrm{max}}`). Then, these harmonics light curves are run
through a principle component analysis (PCA) to determine a set of
orthogonal light curves (\'\'eigencurves\'\'), ordered by
importance. The eigencurves are linearly combined with the uniform-map
stellar and planetary light curves and fit, individually, to the
spectroscopic light curves. That is, the same set of eigencurves are
used for each spectroscopic bin, but they are allowed different
weights. Functionally, the model is the following:

.. math::
   F_{\textrm{sys}}(t) = c_0 Y_0^0 + \sum_i^N c_i E_i + F_{\textrm{star}} + s_{\textrm{corr}},

where :math:`F_{\textrm{sys}}` is the system flux, :math:`N` is the
number of eigencurves to use (set by the user), :math:`c_i` are the
eigencurve weights, :math:`E_i` are the eigencurves, :math:`Y_0^0` is
the light curve of the uniform-map planet, :math:`F_{\textrm{star}}`
is the light curve of the star (likely constant and equal to unity, if
the data are normalized), and :math:`s_{\textrm{corr}}` is a constant
term to correct for any errors in stellar normalization.

This model is fit to each spectroscopic light curve using a
least-squares minimization algorithm. Optionally, the user may specify
that emitted fluxes must be positive. Negative fluxes are problematic
because they are non-physical, and imply a negative temperature, which
will cause problems for any attempts to run radiative transfer, a
necessary step in 3D fitting. If this option is enabled, the model
includes a check for positive fluxes; if negatives are found, the
model returns a very poor fit, forcing the fit toward physically
plausible thermal maps. Note that the code only enforces this
condition on visible grid cells of the planet. Although the flux maps
are defined on non-visible grid cells, this is only due to the
continuity enforced by spherical harmonics. In reality, non-visible
cells are completely unconstrained.

The code then uses the :math:`c_i` weights along with the eigenmaps that
match the eigencurves to compute a single flux map for each input
light curve (Equation 4 of Rauscher et al., 2018):

.. math::
   Z_p(\theta, \phi) = c_0 Y_0^0(\theta, \phi) + \sum_i^Nc_iZ_i(\theta, \phi),

where :math:`Z_p` is the flux map, :math:`\theta` is latitude,
:math:`\phi` is longitude, and :math:`Z_i` are the eigenmaps.

These flux maps are converted to temperature maps using Equation 8 of
Rauscher et al., 2018 (here we have included the stellar correction
term as described in the appendix):

.. math::
   T_p(\theta, \phi) = (hc / \lambda k) / \textrm{ln} \left[1 + \left(\frac{R_p}{R_s}\right)^2 \frac{\textrm{exp}[hc/\lambda k T_s] - 1}{\pi Z_p(\theta, \phi) (1 + s_{\textrm{corr}})}\right],

where :math:`\lambda` is the band-averaged wavelength of the filter
used to observe the related light curve, :math:`R_p` is the radius of
the planet, :math:`R_s` is the radius of the star, and :math:`T_s` is
the stellar temperature.

The Visibility Function
^^^^^^^^^^^^^^^^^^^^^^^

The 2D mode also computes the visibility function, which describes the
visibility of each grid cell on the planet as a function of
time. There are two sources of reduced visibility: line-of-sight
(reduced visibility toward the limb) and the star. Thus,

.. math::
   V(\theta, \phi, t) = L(\theta, \phi, t) S(\theta, \phi, t),

where :math:`V` is the visibility, :math:`L` is line-of-sight
visibility, and :math:`S` is visibility due to the star. We define
:math:`\theta` and :math:`\phi` to be locations on the planet with
respect to the observer. As the planet revolves, the
\'\'sub-observer\'\' point (:math:`\theta = 0`, :math:`\phi = 0`)
moves in true latitude and longitude. The :math:`\theta` and
:math:`\phi` of each grid cell change with time, but the cells'
latitudes and longitudes are constant.

Line-of-sight visibility depends on angular distance from the point on
the planet closest to the observer and the area of the discrete grid
cell, integrated over the visible portion of the grid cell. :math:`L`
is then

.. math::
   L(\theta, \phi, t) = \int_{\theta_i}^{\theta_f}\int_{\phi_i}^{\phi_f} R_p^2 \cos^2\theta\cos\phi d\phi d\theta

where :math:`(\theta_i, \theta_f)` is the range of visible
:math:`\theta` and :math:`(\phi_i, \phi_f)` is the range of visible
:math:`\phi` for each grid cell.

Stellar visibility is the crux of eclipse mapping. As the planet moves
behind the star, it is gradually eclipsed, from west to east, and then
vice versa when the planet reemerges. Different grid cells are visible
at different times, which enables disentangling the emission of each
grid cell from the planetary emission as a whole. Currently, the code
uses a very simple form of stellar visibility, where a grid cell is
flagged as 100% visible or 0% visible depending on its location
projected onto the plane perpendicular to the observer's line of
sight. In functional form,

.. math::
   S(\theta, \phi, t) &= 0 \quad \text{if}\, d < R_s\\
   S(\theta, \phi, t) &= 1 \quad \text{otherwise}

where :math:`d` is the projected distance between the center of the
visible portion of the grid cell and the center of the star, defined
as

.. math::
    d &= \sqrt{(x_{\rm cell} - x_s)^2 + (y_{\rm cell} - y_s)^2} \\
      &= \sqrt{(x_p + R_p \cos\bar\theta\sin\bar\phi - x_s)^2 + (y_p + R_p \sin\bar\theta - y_s)^2}.

:math:`x_p` is the :math:`x` position of the planet, :math:`x_s` is
the :math:`x` position of the star, :math:`y_p` is the :math:`y`
position of the planet, and :math:`y_s` is the :math:`y` position of
the star. :math:`\bar\theta` is the average visible :math:`\theta` and
:math:`\bar\phi` is the average visible :math:`\phi`.

We compute :math:`L` and :math:`S` for every grid cell at every time in the
observation.  Later, in the 3D operating mode, this precomputed
visibility grid is multiplied with the planetary emitted flux and then
summed over the grid cells at each time to compute the spectroscopic
light curves.

3D Mode
-------

The 3D portion of the code places the 2D thermal maps vertically
in the planet's atmosphere, generates an atmospheric composition,
runs radiative transfer on each grid cell, integrates the emergent
flux over the observation filters, combines the flux with the
visibility function, and integrates over the planet to calculate
spectroscopic light curves for comparison to the data. The process
is done thousands to millions of times behind an MCMC algorithm
to accurately estimate parameter uncertainties.

Temperature-Pressure Mapping Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The manner in which the thermal maps are placed vertically in the
atmosphere is one of the most important choices in the 3D model.
The following options are currently available:
   
* Isobaric -- Each 2D thermal map is placed at a single pressure for
  all grid cells. There is one free parameter, a log-pressure level,
  for each thermal map.
* Sinusoidal -- Each 2D thermal map is placed according to a sinusoid,
  in both longitude and latitude. The longitudinal phase can vary.
  Functionally, the model is:

  .. math::
     \log p(\theta, \phi) = a_1 + a_2\cos\theta + a_3\cos(\phi - a_4)

  where :math:`a_i` are free parameters. There are four free parameters
  per thermal map.
* Flexible -- Each visible grid cell of each thermal map has its own
  parameter for its pressure level. The number of free parameters
  depends on the latitude-longitude resolution of the 3D map.


Atmospheric Composition
^^^^^^^^^^^^^^^^^^^^^^^

The code also generates an atmospheric composition, as atomic and
molecular abundances vs.\ pressure for each grid cell. ThERESA
offers two schemes for calculating atmospheric composition:

* `rate <https://github.com/pcubillos/rate>`_ -- Thermochemical
  abundances are computed analytically as needed.
* `GGchem <https://github.com/pw31/GGchem>`_ -- The user supplies a
  file describing thermochemical equilibrium over a range of
  temperatures and pressures, which is then interpolated as needed.

There is no significant difference in runtime between the two. GGchem
requires slightly more work by the user, but is valid over a larger
range of temperatures and pressures. In theory, more complex schemes
are possible, including options to fit to atmospheric composition.


Radiative Transfer
^^^^^^^^^^^^^^^^^^

Once the temperature and compositional structure of the atmosphere are
set, the code runs radiative transfer to calculate the emergent flux
from each grid cell. The following radiative transfer packages are
available:

* `Tau-REx 3 <https://github.com/ucl-exoplanets/TauREx3_public>`_

For the sake of efficiency, radiative transfer is only run on grid
cells which are visible at some point during the observation. This
also prevents problems when negative temperatures are present on the
non-visible portions of the planet. If negative temperatures are found
in any visible grid cells, the code will return negative fluxes, which
should always be a worse fit than any physical fluxes, thereby driving
any fitting or MCMC algorithm toward non-negative temperatures.

The emergent flux from each grid cell is then integrated over the
observation filters and combined with the visibility function to
generate light curves for comparison to the data.


Contribution Function Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ThERESA has an option to enforce some consistency in the 3D model by
penalizing the goodness-of-fit based on how close the 2D thermal maps
are placed, in pressure, to the pressures which contribute to the
emergent flux at the wavelengths corresponding to each thermal map.
This check is done for every observational filter and every visible
grid cell. Without this check enable, ThERESA may find a \'\'good\'\'
fit which is physically implausible. For example, a 2D map may be
buried deep in the atmosphere, where it has no effect on the emergent
spectrum.


MCMC
^^^^

The light-curve model function, described above, is run within an MCMC
to explore parameter space and accurately estimate parameter
uncertainties.  The MCMC is done through `MC3
<https://github.com/pcubillos/MC3>`_, which offers 3 sampling
algorithms: Metropolis-Hastings random walk, Differential Evolutions
Markov-chain Monte Carlo, and \'\'snooker\'\'.
