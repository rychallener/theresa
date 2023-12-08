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
    for d in fit.datasets:
        print(d.wlmid)

    # Create star, planet, and system objects
    # Not added to fit obj because they aren't pickleable
    print("Initializing star and planet objects.")
    star, planet, system = utils.initsystem(fit, 1)

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

    for d in fit.datasets:
        print("Precomputing - {}".format(d.name))

        print("Computing planet and star positions at observation times.")
        d.x, d.y, d.z = [a.eval() for a in system.position(d.t)]

        print("Calculating uniform-map planet and star fluxes.")
        d.sflux, d.pflux_y00 = [a.eval() for a in  \
                                system.flux(d.t, total=False)]

        print("Calculating minimum and maximum observed longitudes.")
        d.minvislon, d.maxvislon = utils.vislon(planet, d)
        print("Minimum Longitude: {:6.2f}".format(d.minvislon))
        print("Maximum Longitude: {:6.2f}".format(d.maxvislon))

        # Indices of visible cells (only considers longitudes)
        ivis = np.where((fit.lon + fit.dlon / 2. > d.minvislon) &
                        (fit.lon - fit.dlon / 2. < d.maxvislon))
        d.ivislat, d.ivislon = ivis
    
    if not os.path.isdir(cfg.twod.outdir):
        os.mkdir(cfg.twod.outdir)

    print("Optimizing 2D maps.")
    for d in fit.datasets:
        d.maps = []
        for i in range(len(d.wlmid)):
            print("{:.2f} um".format(d.wlmid[i]))
            m = fc.Map()
            
            d.maps.append(m)
            
            m.wlmid     = d.wlmid[i]
            m.filtwl    = d.filtwl[i]
            m.filtwn    = d.filtwn[i]
            m.filttrans = d.filttrans[i]
            m.flux      = d.flux[i]
            m.ferr      = d.ferr[i]

            # Where to put wl-specific outputs
            m.subdir = '{}-filt{}'.format(d.name, i+1)
            if not os.path.isdir(os.path.join(cfg.twod.outdir, m.subdir)):
                os.mkdir(os.path.join(cfg.twod.outdir, m.subdir))

            minbic = np.inf

            for l in range(1, cfg.twod.lmax+1):
                for n in range(0, cfg.twod.ncurves+1):
                    # Skip cases where n is higher than the number of
                    # available eigencurves, which is (l+1)**2, minus
                    # the uniform (l=0) case, since that's included by
                    # default
                    if n > (l+1)**2 - 1:
                        continue

                    # Also let's only do the n=0 case once, since
                    # it's exactly the same fit for every lmax.
                    # Link the LN objects for looping simplicity later
                    if l > 1 and n==0:
                        setattr(m, 'l{}n{}'.format(l, n), m.l1n0)
                        continue

                    print("Fitting lmax={}, n={}".format(l,n))
                    setattr(m, 'l{}n{}'.format(l, n), fc.LN())
                    ln = getattr(m, 'l{}n{}'.format(l, n))

                    ln.subdir = 'l{}n{}'.format(l,n)

                    ln.wlmid = d.wlmid[i]

                    ln.ncurves = n
                    ln.lmax    = l

                    # New planet object with updated lmax
                    star, planet, system = utils.initsystem(fit, ln.lmax)

                    print("Running PCA to determine eigencurves.")
                    ncomp = ln.ncurves
                    if ln.ncurves == 0:
                        ncomp = None
                        
                    ln.eigeny, ln.evalues, ln.evectors, ln.ecurves, ln.lcs = \
                        eigen.mkcurves(system, d.t, ln.lmax,
                                       d.pflux_y00, ncurves=ncomp,
                                       method=cfg.twod.pca,
                                       orbcheck=cfg.twod.orbcheck,
                                       sigorb=cfg.twod.sigorb)

                    print("Calculating intensities of visible grid cells of each eigenmap.")
                    ln.intens, ln.vislat, ln.vislon = \
                        eigen.intensities(fit, d, ln)

                    # Set up for MCMC
                    if cfg.twod.posflux:
                        intens = ln.intens
                    else:
                        intens = None

                    params, pstep, pmin, pmax, pnames, texnames, pindex = \
                        model.get_par_2d(fit, d, ln)

                    baselines = np.array([v.baseline for v in d.visits])
                        
                    indparams = (ln.ecurves, d.t, d.pflux_y00, d.sflux,
                                 ln.ncurves, intens, pindex,
                                 baselines,
                                 [v.tloc for v in d.visits])

                    # Better initial guess if possible
                    if hasattr(m, "l{}n{}".format(l,n-1)):
                        params = getattr(m, "l{}n{}".format(l,n-1)).bestp
                        params = np.insert(params, n-1, 0.0)

                    mc3data = d.flux[i]
                    mc3unc  = d.ferr[i]
                    mc3npz = os.path.join(cfg.twod.outdir,
                                          m.subdir,
                                          ln.subdir,
                                          '2dmcmc-l{}n{}-{:.2f}um.npz'.format(
                                              l,
                                              n,
                                              d.wlmid[i]))

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

                    # Insert fixed values into posterior and bestp. Only
                    # does something if you manually fix parameters by
                    # editing code...
                    niter, nfree = ln.post.shape
                    nparams = len(params)
                    for ip in range(nparams):
                        if pstep[ip] == 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * params[ip],
                                axis=1)
                        if pstep[ip] < 0:
                            ln.post = np.insert(
                                ln.post, ip,
                                np.ones(niter) * ln.bestp[-int(pstep[ip])],
                                axis=1)

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
            m.fmappost, m.tmappost = utils.tmappost(fit, m, m.bestln)
            m.tmapunc = np.std(m.tmappost, axis=0)
            m.fmapunc = np.std(m.fmappost, axis=0)

    print("Optimum lmax and ncurves:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  {:.2f} um: lmax={}, ncurves={}".format(m.wlmid,
                                                            m.bestln.lmax,
                                                            m.bestln.ncurves))
        
    # Save stellar correction terms (we need them later)
    #fit.scorr = np.zeros(len(fit.maps))
    #for i in range(len(fit.maps)):
    #    fit.scorr[i] = fit.maps[i].bestln.bestp[fit.maps[i].bestln.ncurves+1]

    print("Calculating planet visibility with time.")
    for d in fit.datasets:
        print(d.name)
        pbar = progressbar.ProgressBar(max_value=len(d.t))
        nt = len(d.t)
        d.vis = np.zeros((nt, cfg.twod.nlat, cfg.twod.nlon))
        for it in range(len(d.t)):
            d.vis[it] = utils.visibility(d.t[it],
                                         np.deg2rad(fit.lat),
                                         np.deg2rad(fit.lon),
                                         np.deg2rad(fit.dlatgrid),
                                         np.deg2rad(fit.dlongrid),
                                         np.deg2rad(180.),
                                         cfg.planet.prot,
                                         cfg.planet.t0,
                                         cfg.planet.r, cfg.star.r,
                                         d.x[:,it], d.y[:,it])
            pbar.update(it+1)

    print("Checking for negative fluxes in visible cells:")
    for d in fit.datasets:
        print(d.name)
        for m in d.maps:
            print("  Wl: {:.2f} um".format(m.wlmid))
            for i in range(m.bestln.intens.shape[1]):
                check = np.sum(m.bestln.intens[:,i] *
                               m.bestln.bestp[:m.bestln.ncurves]) + \
                               m.bestln.bestp[ m.bestln.ncurves] / np.pi
                if check <= 0.0:
                    msg = "    Lat: {:+07.2f}, Lon: {:+07.2f}, Flux: {:+013.10f}"
                    print(msg.format(m.dataset.vislat[i],
                                     m.dataset.vislon[i],
                                     check))

    print("Constructing total flux and brightness temperature maps " +
          "from eigenmaps.")
    for d in fit.datasets:
        for m in d.maps:
            star, planet, system = utils.initsystem(fit, m.bestln.lmax)
            # These are used or not used in mkmaps depending on the type
            # of stellar spectrum set in the configuration.
            fwl    = m.filtwl
            ftrans = m.filttrans
            swl    = fit.starwl
            sspec  = fit.starflux
            fmap, tmap = eigen.mkmaps(planet, m.bestln.eigeny,
                                      m.bestln.bestp,
                                      m.bestln.ncurves, m.wlmid,
                                      cfg.star.r, cfg.planet.r,
                                      cfg.star.t, fit.lat, fit.lon,
                                      starspec=cfg.star.starspec,
                                      fwl=fwl, ftrans=ftrans, swl=swl,
                                      sspec=sspec)
            m.fmap = fmap
            m.tmap = tmap

    print("Temperature ranges of maps:")
    for d in fit.datasets:
        for m in d.maps:
            print("  {:.2f} um:".format(m.wlmid))
            tmax = np.max(m.tmap[~np.isnan(m.tmap)])
            tmin = np.min(m.tmap[~np.isnan(m.tmap)])
            print("    Max: {:.2f} K".format(tmax))
            print("    Min: {:.2f} K".format(tmin))
            print("    Negative: {:f}".format(np.sum(np.isnan(m.tmap))))

    # Make a single array of tmaps for convenience
    fit.nmaps = np.sum([len(d.maps) for d in fit.datasets])
    fit.tmaps = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    fit.fmaps = np.zeros((fit.nmaps, fit.cfg.twod.nlat, fit.cfg.twod.nlon))

    imap = 0
    for d in fit.datasets:
        for m in d.maps:
            fit.tmaps[imap] = m.tmap
            fit.fmaps[imap] = m.fmap
            imap += 1

    if cfg.twod.plots:
        print("Making plots.")
        for d in fit.datasets:
            for m in d.maps:
                outdir = os.path.join(cfg.twod.outdir, m.subdir)
                # Make sure the planet has the right lmax
                star, planet, system = utils.initsystem(fit, m.bestln.lmax)
                plots.emaps(planet, m.bestln.eigeny, outdir, proj='ortho')
                plots.emaps(planet, m.bestln.eigeny, outdir, proj='rect')
                plots.emaps(planet, m.bestln.eigeny, outdir, proj='moll')
                plots.lightcurves(d.t, m.bestln.lcs, outdir)
                plots.eigencurves(d.t, m.bestln.ecurves, outdir,
                                  ncurves=m.bestln.ncurves)
                plots.ecurvepower(m.bestln.evalues, outdir)
            
        plots.pltmaps(fit)
        plots.tmap_unc(fit)
        plots.bestfit(fit)
        plots.ecurveweights(fit)
        plots.hshist(fit)
        plots.bics(fit, outdir=cfg.twod.outdir)

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
        if cfg.cfg.has_option('GGchem', 'dispolfiles'):
            dispolfiles = cfg.cfg.get('GGchem', 'dispolfiles')
        else:
            dispolfiles = None
        
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
                                        elements=cfg.threed.elem,
                                        dispolfiles=dispolfiles)
    else:
        fit.cheminfo = None

    # Determine which grid cells to use
    # Get all unique lat/lon combinations from all datasets
    # Only considers longitudes currently
    allivislon = np.concatenate([d.ivislon for d in fit.datasets])
    allivislat = np.concatenate([d.ivislat for d in fit.datasets])
    allivis = [(a,b) for a, b in zip(allivislat, allivislon)]
    allivis = np.unique(allivis, axis=1)
    fit.ivislat3d = np.array([a[0] for a in allivis])
    fit.ivislon3d = np.array([a[1] for a in allivis])
        
    print("Fitting spectrum.")
    # This doesn't work. Stick to TauREx.
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

        for m in fit.maps:
            filtwn = m.filtwn
            filttrans = m.filttrans
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
        mc3data   = np.concatenate([m.flux for m in fit.maps])
        mc3uncert = np.concatenate([m.ferr for m in fit.maps])
        
        if cfg.threed.fitcf:
            ncfpar = np.sum([
                d.ivislat.size * len(d.wlmid) for d in fit.datasets])
            print("ncf: " + str(ncfpar))
            #ncfpar = fit.ivislat.size * len(cfg.twod.filtfiles)
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
          
    nmaps = len(fit.maps)

    print("Calculating best fit.")
    specout = model.specgrid(fit.specbestp, fit)
    fit.fluxgrid    = specout[0]
    fit.besttgrid   = specout[1]
    fit.taugrid     = specout[2]
    fit.p           = specout[3]
    fit.modelwngrid = specout[4]
    fit.pmaps       = specout[5]
    
    fit.specbestmodel = model.sysflux(fit.specbestp, fit)[0]
    #fit.specbestmodel = fit.specbestmodel.reshape((nfilt, nt))

    allmols = np.concatenate((cfg.threed.mols, cfg.threed.cmols))
    print("WARNING: assuming solar metallicity for plotting (fix this)!")
    fit.abnbest, fit.abnspec = atm.atminit(fit.cfg.threed.atmtype,
                                           allmols, fit.p, fit.besttgrid,
                                           cfg.planet.m, cfg.planet.r,
                                           cfg.planet.p0, 0.0,
                                           ilat=fit.ivislat3d,
                                           ilon=fit.ivislon3d,
                                           cheminfo=fit.cheminfo)
                                           

    print("Calculating contribution functions.")
    allfiltwn    = [m.filtwn    for m in fit.maps]
    allfilttrans = [m.filttrans for m in fit.maps]
    fit.cf = cf.contribution_filters(fit.besttgrid, fit.modelwngrid,
                                     fit.taugrid, fit.p, allfiltwn,
                                     allfilttrans)

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
        plots.spectra(fit, outdir=outdir)
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
        
    
        

    

    
