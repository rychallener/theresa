# ThERESA v. 1.1

## New Functionality

	- 2D mapping now optimizes the model for the user. The lmax and ncurves configuration options now specify maxima for these values, and ThERESA will find the optimal combination for each wavelength bin.

	- You can now fit multiple asynchronous datasets over multiple instruments. The configuration file has been restructured into "Instrument" and "Observation" sections, and each observation is tied to an instrument. All observation with the same instrument will be fit with the same 2D map.

	- The 3D grid is now optimized to better sample the planet, with higher spatial resolution near the equator and lower at the poles. Furthermore, the visibility of each grid cell in time is calculated analytically, which is both faster and more accurate, particularly for lower spatial resolution. This results in a ~4x decrease in runtime for 3D retrievals.

	- The systematic-model fitting has been overhauled. Each observation can now be fit with its own baseline (e.g., linear, exponential, etc.) model, and ThERESA now support detrending against arbitrary time-dependent vectors like spectral trace position and width.

	- You can now fit for a separate normalization factor for each observation, which allows fitting for long-term stellar variability.

## Major Updates

	- 3D fits can now be resumed from where they ended.

	- New plots!

	- Better initial guesses for 2D mapping for faster convergence.

	- 2D fits now exit MCMC once they have converged.

	- 2D mapping includes a uniform-planet fit (ncurves = 0).

	- Account for light travel-time delay in fitting.

## Minor Updates

	- Improvements to runtime in both 2D and 3D fitting.

