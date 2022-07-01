#! /usr/bin/env python3

# General imports
import os
import sys
import mc3
import pickle
import starry
import shutil
import subprocess
import progressbar
import numpy as np
import matplotlib.pyplot as plt

# Taurex imports
import taurex
from taurex import chemistry
from taurex import planet
from taurex import stellar
from taurex import model
from taurex import pressure
from taurex import temperature
from taurex import cache
from taurex import contributions
from taurex import optimizer
# This import is explicit because it's not included in taurex.temperature. Bug?
from taurex.data.profiles.temperature.temparray import TemperatureArray

# Taurex is a bit...talkative
import taurex.log
taurex.log.disableLogging()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')
moddir     = os.path.join(libdir,  'modules')
ratedir    = os.path.join(moddir,  'rate')
transitdir = os.path.join(moddir, 'transit')

# Lib imports
sys.path.append(libdir)
import cf
import atm
import pca
import eigen
import model
import plots
import mkcfg
import utils
import constants   as c
import fitclass    as fc
import taurexclass as trc

starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)

def map2d(cfile):
    """
    One function to rule them all.
    """
    # Create the master fit object
    fit = fc.Fit()
    
    print("Reading the configuration file.")
    fit.read_config(cfile)
    cfg = fit.cfg

    print("Reading the data.")
    fit.read_data()

    print("Reading filters.")
    fit.read_filters()
    print("Filter mean wavelengths (um):")
    print(fit.wlmid)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star, planet, system = utils.initsystem(fit, 1)

    print("Computing planet and star positions at observation times.")
    fit.x, fit.y, fit.z = [a.eval() for a in system.position(fit.t)]

    print("Calculating uniform-map planet and star fluxes.")
    fit.sflux, fit.pflux_y00 = [a.eval() for a in  \
                                system.flux(fit.t, total=False)]

    print("Calculating minimum and maximum observed longitudes.")
    fit.minvislon, fit.maxvislon = utils.vislon(planet, fit)
    print("Minimum Longitude: {:6.2f}".format(fit.minvislon))
    print("Maximum Longitude: {:6.2f}".format(fit.maxvislon))

    print("Calculating latitude and longitude of planetary grid.")
    fit.dlat = 180. / cfg.twod.nlat
    fit.dlon = 360. / cfg.twod.nlon
    fit.lat, fit.lon = np.meshgrid(np.linspace(-90  + fit.dlat / 2.,
                                                90  - fit.dlat / 2.,
                                               cfg.twod.nlat, endpoint=True),
                                   np.linspace(-180 + fit.dlon / 2.,
                                                180 - fit.dlon / 2.,
                                               cfg.twod.nlon, endpoint=True),
                                   indexing='ij')
    fit.dlatgrid, fit.dlongrid = np.meshgrid(np.ones(cfg.twod.nlat) * fit.dlat,
                                             np.ones(cfg.twod.nlon) * fit.dlon,
                                             indexing='ij')

    # Indices of visible cells (only considers longitudes)
    ivis = np.where((fit.lon + fit.dlon / 2. > fit.minvislon) &
                    (fit.lon - fit.dlon / 2. < fit.maxvislon))
    fit.ivislat, fit.ivislon = ivis
    
    if not os.path.isdir(cfg.twod.outdir):
        os.mkdir(cfg.twod.outdir)

    # List of 2d map fits
    fit.maps = []

    print("Optimizing 2D maps.")
    for i in range(len(fit.wlmid)):
        print("{:.2f} um".format(fit.wlmid[i]))
        fit.maps.append(fc.Map())

        m = fit.maps[i]
        m.wlmid = fit.wlmid[i]

        # Where to put wl-specific outputs
        m.subdir = 'filt{}'.format(i+1)
        if not os.path.isdir(os.path.join(cfg.twod.outdir, m.subdir)):
            os.mkdir(os.path.join(cfg.twod.outdir, m.subdir))

        minbic = np.inf

        for l in range(1, cfg.twod.lmax[i]+1):
            for n in range(1, cfg.twod.ncurves[i]+1):
                # Skip cases where n is higher than the number of
                # available eigencurves, which is (l+1)**2, minus
                # the uniform (l=0) case, since that's included by
                # default
                if n > (l+1)**2 - 1:
                    continue
                
                print("Fitting lmax={}, n={}".format(l,n))
                setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                ln = getattr(m, 'l{}n{}'.format(l, n))

                ln.subdir = 'l{}n{}'.format(l,n)

                ln.wlmid = fit.wlmid[i]
                
                ln.ncurves = n
                ln.lmax    = l

                # New planet object with updated lmax
                star, planet, system = utils.initsystem(fit, ln.lmax)

                print("Running PCA to determine eigencurves.")
                ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                    eigen.mkcurves(system, fit.t, ln.lmax,
                                   fit.pflux_y00, ncurves=ln.ncurves,
                                   method=cfg.twod.pca)

                print("Calculating intensities of visible grid cells of each eigenmap.")
                ln.intens, ln.vislat, ln.vislon = eigen.intensities(fit, ln)

                # Set up for MCMC
                if cfg.twod.posflux:
                    intens = ln.intens
                else:
                    intens = None
        
                indparams = (ln.ecurves, fit.t, fit.pflux_y00, fit.sflux,
                             ln.ncurves, intens, cfg.twod.baseline)

                params, pstep, pmin, pmax, pnames, texnames = \
                    model.get_par_2d(fit, ln)

                mc3data = fit.flux[i]
                mc3unc  = fit.ferr[i]
                mc3npz = os.path.join(cfg.twod.outdir,
                                      m.subdir,
                                      ln.subdir,
                                      '2dmcmc-l{}n{}-{:.2f}um.npz'.format(
                                          l,
                                          n,
                                          fit.wlmid[i]))

                mc3out = mc3.sample(data=mc3data, uncert=mc3unc,
                                    func=model.fit_2d,
                                    nsamples=cfg.twod.nsamples,
                                    burnin=cfg.twod.burnin,
                                    ncpu=cfg.twod.ncpu, sampler='snooker',
                                    savefile=mc3npz, params=params,
                                    indparams=indparams, pstep=pstep,
                                    leastsq=cfg.twod.leastsq,
                                    plots=cfg.twod.plots, pmin=pmin,
                                    pmax=pmax, pnames=pnames,
                                    texnames=texnames, thinning=10,
                                    fgamma=cfg.twod.fgamma)

                # MC3 doesn't clear its plots >:(
                plt.close('all')

                ln.bestfit = mc3out['best_model']
                ln.bestp   = mc3out['bestp']
                ln.stdp    = mc3out['stdp']
                ln.chisq   = mc3out['best_chisq']
                ln.post    = mc3out['posterior']
                ln.zmask   = mc3out['zmask']

                ln.nfreep = np.sum(pstep > 0)
                ln.ndata  = mc3data.size

                ln.redchisq = ln.chisq / \
                    (ln.ndata - ln.nfreep)
                ln.bic      = ln.chisq + \
                    ln.nfreep * np.log(ln.ndata)

                print("Chisq:         {}".format(ln.chisq))
                print("Reduced Chisq: {}".format(ln.redchisq))
                print("BIC:           {}".format(ln.bic))

                if ln.bic < minbic:
                    minbic = ln.bic
                    m.bestln = ln

        print("Calculating hotspot latitude and longitude.")
        hs = utils.hotspotloc_driver(fit, m.bestln)
        m.hslocbest  = hs[0]
        m.hslocstd   = hs[1]
        m.hslocpost  = hs[2]
        m.hsloctserr = hs[3]

        msg = "Hotspot Longitude: {:.2f} +{:.2f} {:.2f}"
        print(msg.format(m.hslocbest[1],
                         m.hsloctserr[1][0],
                         m.hsloctserr[1][1]))

        print("Calculating temperature map uncertainties.")
        m.fmappost, m.tmappost = utils.tmappost(fit, m.bestln)
        m.tmapunc = np.std(m.tmappost, axis=0)
        m.fmapunc = np.std(m.fmappost, axis=0)

    # Useful prints
    fit.totchisq2d    = np.sum([m.bestln.chisq for m in fit.maps])
    fit.totredchisq2d = fit.totchisq2d / \
        (np.sum([(m.bestln.ndata - m.bestln.nfreep) for m in fit.maps]))
    fit.totbic2d      = np.sum([m.bestln.bic for m in fit.maps])
    print("Total Chisq:         {}".format(fit.totchisq2d))
    print("Total Reduced Chisq: {}".format(fit.totredchisq2d))
    print("Total BIC:           {}".format(fit.totbic2d))

    print("Optimum lmax and ncurves:")
    for m in fit.maps:
        print("  {:.2f} um: lmax={}, ncurves={}".format(m.wlmid,
                                                        m.bestln.lmax,
                                                        m.bestln.ncurves))
        
    # Save stellar correction terms (we need them later)
    fit.scorr = np.zeros(len(fit.wlmid))
    for i in range(len(fit.wlmid)):
        fit.scorr[i] = fit.maps[i].bestln.bestp[fit.maps[i].bestln.ncurves+1]

    print("Calculating planet visibility with time.")
    pbar = progressbar.ProgressBar(max_value=len(fit.t))
    nt = len(fit.t)
    fit.vis = np.zeros((nt, cfg.twod.nlat, cfg.twod.nlon))
    for it in range(len(fit.t)):
        fit.vis[it] = utils.visibility(fit.t[it],
                                       np.deg2rad(fit.lat),
                                       np.deg2rad(fit.lon),
                                       np.deg2rad(fit.dlatgrid),
                                       np.deg2rad(fit.dlongrid),
                                       np.deg2rad(180.),
                                       cfg.planet.prot,
                                       cfg.planet.t0,
                                       cfg.planet.r,
                                       cfg.star.r,
                                       fit.x[:,it], fit.y[:,it])
        pbar.update(it+1)

    print("Checking for negative fluxes in visible cells:")
    for j in range(len(fit.wlmid)):
        print("  Wl: {:.2f} um".format(fit.wlmid[j]))
        m = fit.maps[j]
        for i in range(m.bestln.intens.shape[1]):
            check = np.sum(m.bestln.intens[:,i] *
                           m.bestln.bestp[:m.bestln.ncurves]) + \
                           m.bestln.bestp[ m.bestln.ncurves] / np.pi
            if check <= 0.0:
                msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                print(msg.format(fit.vislat[i], fit.vislon[i], check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    for j in range(len(fit.wlmid)):
        star, planet, system = utils.initsystem(fit, fit.maps[j].bestln.lmax)
        # These are used or not used in mkmaps depending on the type
        # of stellar spectrum set in the configuration.
        fwl    = fit.filtwl[j]
        ftrans = fit.filttrans[j]
        swl    = fit.starwl
        sspec  = fit.starflux
        fmap, tmap = eigen.mkmaps(planet, fit.maps[j].bestln.eigeny,
                                  fit.maps[j].bestln.bestp,
                                  fit.maps[j].bestln.ncurves,
                                  fit.wlmid[j], cfg.star.r,
                                  cfg.planet.r, cfg.star.t, fit.lat,
                                  fit.lon, starspec=cfg.star.starspec,
                                  fwl=fwl, ftrans=ftrans, swl=swl,
                                  sspec=sspec)
        fit.maps[j].fmap = fmap
        fit.maps[j].tmap = tmap

    print("Temperature ranges of maps:")
    for i in range(len(fit.wlmid)):
        print("  {:.2f} um:".format(fit.wlmid[i]))
        tmax = np.max(fit.maps[i].tmap[~np.isnan(fit.maps[i].tmap)])
        tmin = np.min(fit.maps[i].tmap[~np.isnan(fit.maps[i].tmap)])
        print("    Max: {:.2f} K".format(tmax))
        print("    Min: {:.2f} K".format(tmin))
        print("    Negative: {:f}".format(np.sum(np.isnan(fit.maps[i].tmap))))

    # Make a single array of tmaps for convenience
    fit.tmaps = np.array([m.tmap for m in fit.maps])
    fit.fmaps = np.array([m.fmap for m in fit.maps])

    if cfg.twod.plots:
        print("Making plots.")
        for m in fit.maps:
            outdir = os.path.join(cfg.twod.outdir, m.subdir)
            # Make sure the planet has the right lmax
            star, planet, system = utils.initsystem(fit, m.bestln.lmax)
            plots.emaps(planet, m.bestln.eigeny, outdir, proj='ortho')
            plots.emaps(planet, m.bestln.eigeny, outdir, proj='rect')
            plots.emaps(planet, m.bestln.eigeny, outdir, proj='moll')
            plots.lightcurves(fit.t, m.bestln.lcs, outdir)
            plots.eigencurves(fit.t, m.bestln.ecurves, outdir,
                              ncurves=m.bestln.ncurves)
            plots.ecurvepower(m.bestln.evalues, outdir)
            
        plots.pltmaps(fit)
        plots.tmap_unc(fit)
        plots.bestfit(fit)
        plots.ecurveweights(fit)
        plots.hshist(fit)

    if cfg.twod.animations:
        print("Making animations.")
        plots.visanimation(fit, outdir=cfg.twod.outdir)
        plots.fluxmapanimation(fit, outdir=cfg.twod.outdir)

    fit.save(cfg.twod.outdir)

def map3d(fit, system):
    cfg = fit.cfg
    outdir = os.path.join(cfg.threed.indir, cfg.threed.outdir)
    # Handle any atmosphere setup
    if cfg.threed.atmtype == 'ggchem':
        print("Precomputing chemistry grid.")
        # T, P, z, spec, abn
        fit.cheminfo = atm.setup_GGchem(cfg.threed.tmin,
                                        cfg.threed.tmax,
                                        cfg.threed.numt,
                                        cfg.threed.ptop,
                                        cfg.threed.pbot,
                                        cfg.threed.nlayers,
                                        cfg.threed.zmin,
                                        cfg.threed.zmax,
                                        cfg.threed.numz,
                                        condensates=cfg.threed.condensates,
                                        elements=cfg.threed.elem)
    else:
        fit.cheminfo = None
        
    print("Fitting spectrum.")
    if cfg.threed.rtfunc == 'transit':
        tcfg = mkcfg.mktransit(cfile, outdir)
        rtcall = os.path.join(transitdir, 'transit', 'transit')
        opacfile = cfg.cfg.get('transit', 'opacityfile')
        if not os.path.isfile(opacfile):
            print("  Generating opacity grid.")
            subprocess.call(["{:s} -c {:s} --justOpacity".format(rtcall, tcfg)],
                            shell=True, cwd=outdir)
        else:
            print("  Copying opacity grid: {}".format(opacfile))
            try:
                shutil.copy2(opacfile, os.path.join(outdir,
                                                    os.path.basename(opacfile)))
            except shutil.SameFileError:
                print("  Files match. Skipping.")
                pass
        subprocess.call(["{:s} -c {:s}".format(rtcall, tcfg)],
                        shell=True, cwd=outdir)

        wl, flux = np.loadtxt(os.path.join(outdir,
                                           cfg.cfg.get('transit', 'outspec')),
                              unpack=True)

    elif cfg.threed.rtfunc == 'taurex':
        # Make sure the wn range is appropriate
        wnlow  = cfg.cfg.getfloat('taurex', 'wnlow')
        wnhigh = cfg.cfg.getfloat('taurex', 'wnhigh')
        wndelt = 1.0
        
        for filtwn, filttrans in zip(fit.filtwn, fit.filttrans):
            nonzero = filtwn[np.where(filttrans != 0.0)]
            if not np.all((nonzero > wnlow) & (nonzero < wnhigh)):
                print("ERROR: Wavenumber range does not cover all filters!")
                sys.exit()
                
        fit.wngrid = np.arange(wnlow, wnhigh, wndelt)

        # Note: must do these things in the right order
        taurex.cache.OpacityCache().clear_cache()
        taurex.cache.OpacityCache().set_opacity_path(cfg.cfg.get('taurex',
                                                                 'csxdir'))
        taurex.cache.CIACache().set_cia_path(cfg.cfg.get('taurex',
                                                         'ciadir'))

        indparams = [fit]

        # Get sensible defaults
        params, pstep, pmin, pmax, pnames, nparams, modeltype, imodel = \
            model.get_par_3d(fit)

        fit.nparams3d   = nparams
        fit.modeltype3d = modeltype
        fit.imodel3d    = imodel

        # Override if specified by the user
        if hasattr(cfg.threed, 'params'):
            params = cfg.threed.params
        if hasattr(cfg.threed, 'pmin'):
            pmin   = cfg.threed.pmin
        if hasattr(cfg.threed, 'pmax'):
            pmax   = cfg.threed.pmax
        if hasattr(cfg.threed, 'pstep'):
            pstep  = cfg.threed.pstep
        if hasattr(cfg.threed, 'pnames'):
            pnames = cfg.threed.pnames

        nparams = len(params)

        mc3npz = os.path.join(outdir, '3dmcmc.npz')
        

        # Build data and uncert arrays for mc3
        mc3data   = fit.flux.flatten()
        mc3uncert = fit.ferr.flatten()
        if cfg.threed.fitcf:
            ncfpar = fit.ivislat.size * len(cfg.twod.filtfiles)
            # Here we use 0s and 1s for the cf data and uncs, then
            # have the model return a value equal to the number
            # of sigma away from the cf peak, so MC3 computes the
            # correct chisq contribution from each cf
            cfdata = np.zeros(ncfpar)
            cfunc  = np.ones( ncfpar)
            mc3data   = np.concatenate((mc3data,   cfdata))
            mc3uncert = np.concatenate((mc3uncert, cfunc))

        out = mc3.sample(data=mc3data, uncert=mc3uncert,
                         func=model.mcmc_wrapper,
                         nsamples=cfg.threed.nsamples,
                         burnin=cfg.threed.burnin,
                         ncpu=cfg.threed.ncpu, sampler='snooker',
                         savefile=mc3npz, params=params,
                         indparams=indparams, pstep=pstep, pmin=pmin,
                         pmax=pmax, pnames=pnames,
                         leastsq=cfg.threed.leastsq,
                         grbreak=cfg.threed.grbreak,
                         fgamma=cfg.threed.fgamma,
                         plots=cfg.threed.plots,
                         resume=cfg.threed.resume)

        # MC3 doesn't clear its plots >:(
        plt.close('all')

    fit.specbestp  = out['bestp']
    fit.chisq3d    = out['best_chisq']
    fit.redchisq3d = out['red_chisq']
    fit.bic3d      = out['BIC']
    fit.zmask3d    = out['zmask']
    fit.zchain3d   = out['zchain']

    # Put fixed and shared params in the posterior so it's a
    # consistent size
    fit.posterior3d = out['posterior']
    niter, nfree = fit.posterior3d.shape
    for i in range(nparams):
        if pstep[i] == 0:
            fit.posterior3d = np.insert(
                fit.posterior3d, i,
                np.ones(niter) * params[i],
                axis=1)
        if pstep[i] < 0:
            fit.posterior3d = np.insert(
                fit.posterior3d, i,
                np.ones(niter) * fit.specbestp[-int(pstep[i])],
                axis=1)

    # Evaluate SPEIS, ESS, and CR error
    print("Calculating effective sample size.")
    nchains = np.max(fit.zchain3d) + 1
    fit.cspeis3d = np.zeros((nchains, nparams)) # SPEIS by chain
    fit.cess3d   = np.zeros((nchains, nparams)) # ESS by chain
    for i in range(nchains):
        where = np.where(fit.zchain3d[fit.zmask3d] == i)
        chain = fit.posterior3d[fit.zmask3d][where]
        if len(chain) == 0:
            print('WARNING: Chain {} has no accepted iterations!'.format(i))
        else:
            fit.cspeis3d[i], fit.cess3d[i] = utils.ess(chain)

    fit.ess3d   = np.sum(fit.cess3d, axis=0) # Overall ESS
    fit.speis3d = np.ceil(niter / fit.ess3d).astype(int) # Overall SPEIS
    fit.crsig3d = np.zeros(nparams)
    for i in range(nparams):
        fit.crsig3d[i] = utils.crsig(fit.ess3d[i])

    print("\nParameter        SPEIS     ESS   68.3% Error"
          "\n-------------- ------- ------- -------------")
    for i in range(nparams):
        print(f"{pnames[i]:<14s} " +
              f"{fit.speis3d[i]:7d} " +
              f"{fit.ess3d[i]:7.1f} " +
              f"{fit.crsig3d[i]:13.2e}")
          
    nfilt = len(cfg.twod.filtfiles)
    nt    = len(fit.t)

    print("Calculating best fit.")
    specout = model.specgrid(fit.specbestp, fit)
    fit.fluxgrid    = specout[0]
    fit.besttgrid   = specout[1]
    fit.taugrid     = specout[2]
    fit.p           = specout[3]
    fit.modelwngrid = specout[4]
    fit.pmaps       = specout[5]
    
    fit.specbestmodel = model.sysflux(fit.specbestp, fit)[0]
    fit.specbestmodel = fit.specbestmodel.reshape((nfilt, nt))

    allmols = np.concatenate((cfg.threed.mols, cfg.threed.cmols))
    print("WARNING: assuming solar metallicity for plotting (fix this)!")
    fit.abnbest, fit.abnspec = atm.atminit(fit.cfg.threed.atmtype,
                                           allmols, fit.p, fit.besttgrid,
                                           cfg.planet.m, cfg.planet.r,
                                           cfg.planet.p0, 0.0,
                                           ilat=fit.ivislat,
                                           ilon=fit.ivislon,
                                           cheminfo=fit.cheminfo)
                                           

    print("Calculating contribution functions.")
    fit.cf = cf.contribution_filters(fit.besttgrid, fit.modelwngrid,
                                     fit.taugrid, fit.p, fit.filtwn,
                                     fit.filttrans)

    # Save before plots, in case of crashes
    # Do not add attributes to fit after this
    fit.save(outdir)
    
    if cfg.threed.plots:
        plots.bestfitlcsspec(fit, outdir=outdir)
        plots.bestfittgrid(fit, outdir=outdir)
        plots.tau(fit, outdir=outdir)
        plots.pmaps3d(fit, outdir=outdir)
        plots.tgrid_unc(fit, outdir=outdir)
        plots.cf_by_location(fit, outdir=outdir)
        plots.cf_by_filter(fit, outdir=outdir)
        plots.cf_slice(fit, outdir=outdir)
        if 'clouds' in fit.modeltype3d:
            plots.clouds(fit, outdir=outdir)

    if cfg.threed.animations:
        plots.pmaps3d(fit, animate=True, outdir=outdir)

        
if __name__ == "__main__":
    print("#########################################################")
    print("  ThERESA: Three-dimensional Exoplanet Retrieval from    ")
    print("           Eclipse Spectroscopy of Atmospheres           ")
    print("  Copyright 2021-2022 Ryan C. Challener & collaborators  ")
    print("#########################################################")
    
    if len(sys.argv) < 3:
        print("ERROR: Call structure is theresa.py <mode> <configuration file>.")
        sys.exit()
    else:
        mode  = sys.argv[1]
        cfile = sys.argv[2]

    if mode in ['2d', '2D']:
        map2d(cfile)
    elif mode in ['3d', '3D']:
        # Read config to find location of output, load output,
        # then read config again to get any changes from 2d run.
        fit = fc.Fit()
        fit.read_config(cfile)
        fit = fc.load(outdir=fit.cfg.threed.indir)
        fit.read_config(cfile)
        # 3D mapping doesn't care about the degree of harmonics, so
        # just use 1
        star, planet, system = utils.initsystem(fit, 1)
        map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
        
    
        

    

    
