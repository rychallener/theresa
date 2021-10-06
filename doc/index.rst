.. ThERESA documentation master file, created by
   sphinx-quickstart on Thu Jul 22 15:57:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================================================================
ThERESA: Three-dimensional Exoplanet Retrieval from Eclipse Spectroscopy of Atmopsheres
=======================================================================================

``ThERESA`` is a Python package for three-dimensional exoplanet
retrieval from spectroscopic eclipse observations.  It uses principle
component analysis to fit a two-dimensional thermal map to light
curves at each wavelength, or filter, then combines these maps into a
three-dimensional atmosphere, computes emission across the planet
using radiative transfer, and integrates over the planet to calculate
light curves for comparison with observations.

:Author: Ryan Challener and Emily Rauscher
:Contact: rchallen@umich.edu

Documentation
=============
	  
.. toctree::
   :maxdepth: 3

   installation
   gettingstarted
   configuration
   codedesc
   scripts
