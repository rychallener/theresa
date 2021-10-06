#! /usr/bin/env python

import os
import sys
sys.path.append('../lib')
import fortranfilefinisher as fff
import atm
import taurexclass as trc
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['axes.formatter.useoffset'] = True
import numpy as np
import starry
import progressbar
import scipy.interpolate as sci
import scipy.constants as scc
import configparser as cp

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

taurex.log.disableLogging()

cfile = sys.argv[1]

cfg = cp.ConfigParser()
cfg.read(cfile)

planetname = cfg.get('synthlc', 'planetname')
fortfile   = cfg.get('synthlc', 'gcmfile')
outdir     = cfg.get('synthlc', 'outdir')

if not os.path.isdir(outdir):
     os.mkdir(outdir)

# http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
# Retrieved September 10, 2020
Mjup = 1.89819e27 # kg
Rjup = 6.9911e7 # m (volumetric mean)

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
# Retrieved September 11, 2020
Msun = 1.9885e30 # kg
Rsun = 6.957e8   # m (volumetric mean)

# Star parameters
ms = cfg.getfloat('synthlc', 'ms') * Msun
rs = cfg.getfloat('synthlc', 'rs') * Rsun
ts = cfg.getfloat('synthlc', 'ts')
ds = cfg.getfloat('synthlc', 'ds')
zs = cfg.getfloat('synthlc', 'zs')

# Planet parameters
mp   = cfg.getfloat('synthlc', 'mp') * Mjup
rp   = cfg.getfloat('synthlc', 'rp') * Rjup
ap   = cfg.getfloat('synthlc', 'ap')
bp   = cfg.getfloat('synthlc', 'bp')
porb = cfg.getfloat('synthlc', 'porb')
prot = cfg.getfloat('synthlc', 'prot')
t0   = cfg.getfloat('synthlc', 't0')
ecc  = cfg.getfloat('synthlc', 'ecc')
inc  = cfg.getfloat('synthlc', 'inc')

# Atmosphere parameters
atmtype = cfg.get('synthlc', 'atmtype')
atmfile = cfg.get('synthlc', 'atmfile')
mols    = cfg.get('synthlc', 'mols').split()
opacdir = cfg.get('synthlc', 'opacdir')
ciadir  = cfg.get('synthlc', 'ciadir')

# Observation parameters
pstart = cfg.getfloat('synthlc', 'phasestart')
pend   = cfg.getfloat('synthlc', 'phaseend')
dt     = cfg.getfloat('synthlc', 'dt')
necl   = cfg.getint('synthlc', 'necl')
noise  = np.array([float(a) for a in cfg.get('synthlc', 'noise').split()])
filtdir = cfg.get('synthlc', 'filtdir')
filters = [os.path.join(filtdir, a) for a in \
           cfg.get('synthlc', 'filters').split()]

# GCM parameters
path = '.'
runname = ''
oom =   cfg.getfloat('synthlc', 'oom')
surfp = cfg.getfloat('synthlc', 'surfp')
topp  = 10.**(np.log10(surfp) - oom)
ver = False
savet = False

taurex.cache.OpacityCache().clear_cache()
taurex.cache.OpacityCache().set_opacity_path(opacdir)
taurex.cache.CIACache().set_cia_path(ciadir)

prange = (pstart, pend)
t = np.arange(prange[0]*porb,
              prange[1]*porb,
              dt/86400.)
nt = len(t)

# Read filters
filtwl, filtwn, filttrans, wnmid, wlmid = utils.readfilters(filters)
wllo = np.min([np.min(a) for a in filtwl])
wlhi = np.max([np.max(a) for a in filtwl])
wnlo = 10000 / wlhi
wnhi = 10000 / wllo
wngrid = np.linspace(wnlo, wnhi, 100000)

_, oom, surfp, lon, lat, p, d = fff.fort26(path, runname, oom, surfp, ver,
                                           savet, fortfile)

nlev, nlon, nlat, npar = d.shape

# Longitudes are evenly distributed
dlon = np.ones(len(lon)) * np.abs(lon[1] - lon[0])

# Latitudes are not evenly distributed
# This may be not quite right, but at least it sums to 180 degrees
lat_edges = np.zeros(len(lat) + 1)
lat_edges[ 0] =  90
lat_edges[-1] = -90
for i in range(1, len(lat_edges) -1):
     lat_edges[i] = (lat[i] + lat[i-1]) / 2.

dlat = -1 * np.diff(lat_edges)

# Want longitude from -180 to 180 (vis function assumes -90 to 90 is visible)
lon = (lon + 180.) % 360. - 180.

tgrid = np.zeros((nlev, nlat, nlon))

# Create temperature grid in the format atminit expects
for i in range(nlev):
     for j in range(nlat):
          for k in range(nlon):
               tgrid[i,j,k] = d[i,k,j,5]

# Equilibrium chemistry
# (elemfile and refpress are unimportant for this application, but are
#  required inputs)
elemfile = '/home/rchallen/ast/3dmap/code/3dmap/3dmap/inputs/abundances_Asplund2009.txt'
refpress = 0.1
cheminfo = atm.read_GGchem(atmfile)
abn, spec = atm.atminit(atmtype, mols, p, tgrid, mp/Msun, rp/Rsun,
                        0.1, elemfile, '.', cheminfo=cheminfo)

# Planet
rtplan = taurex.planet.Planet(
     planet_mass=mp/Mjup,
     planet_radius=rp/Rjup,
     planet_distance=ap,
     impact_param=bp,
     orbital_period=porb,
     transit_time=t0)
rtstar = taurex.stellar.Star(
     temperature=ts,
     radius=rs/Rsun,
     distance=ds,
     metallicity=zs)
# This is a log(p) profile, same as the GCM output
rtp = taurex.pressure.SimplePressureProfile(nlev, topp*1e5, surfp*1e5) # pascals

fluxgrid = np.empty((nlat, nlon), dtype=list)
taugrid  = np.empty((nlat, nlon), dtype=list)

# Run radiative transfer
print("Running radiative transfer.")
pbar = progressbar.ProgressBar(max_value=nlat*nlon)
for i in range(nlat):
     for j in range(nlon):
          rtt = TemperatureArray(tgrid[:,i,j][::-1])
          rtchem = taurex.chemistry.TaurexChemistry()
          for k in range(len(spec)):
               if (spec[k] in mols):
                    gas = trc.ArrayGas(spec[k], abn[k,:,i,j][::-1])
                    rtchem.addGas(gas)
          rt = trc.EmissionModel3D(
               planet=rtplan,
               star=rtstar,
               pressure_profile=rtp,
               temperature_profile=rtt,
               chemistry=rtchem)
          rt.add_contribution(taurex.contributions.AbsorptionContribution())
          rt.add_contribution(taurex.contributions.CIAContribution())

          rt.build()

          wn, flux, tau, ex = rt.model(wngrid=wngrid)

          fluxgrid[i,j] = flux
          taugrid[i,j] = tau

          pbar.update(i*nlon+j)

nwn = len(wn)
fluxgrid = np.concatenate(np.concatenate(fluxgrid)).reshape(nlat, nlon, nwn)

# Integrate over filters
nfilt = len(filters)
intfluxgrid = np.zeros((nlat, nlon, nfilt))

for i in range(nlat):
     for j in range(nlon):
          intfluxgrid[i,j] = utils.specint(wn, fluxgrid[i,j], filtwn,
                                           filttrans)

# Calculate visibility
planet = starry.kepler.Secondary(starry.Map(ydeg=1),
                                 r=rp/Rsun,
                                 m=mp/Msun,
                                 porb=porb,
                                 prot=porb,
                                 t0=t0,
                                 ecc=ecc,
                                 inc=inc,
                                 theta0=180)

star = starry.kepler.Primary(starry.Map(ydeg=1),
                             r=rs/Rsun,
                             m=ms/Msun)

system = starry.kepler.System(star, planet)

pos = system.position(t)
x = pos[0].eval()
y = pos[1].eval()
z = pos[2].eval()

vis = np.zeros((nt, nlat, nlon))

latgrid, longrid   = np.meshgrid( lat,  lon, indexing='ij')
dlatgrid, dlongrid = np.meshgrid(dlat, dlon, indexing='ij')

print("Calculating visibility.")
pbar = progressbar.ProgressBar(max_value=nt)
for i in range(len(t)):
     vis[i] = utils.visibility(t[i],
                               np.deg2rad(latgrid),
                               np.deg2rad(longrid),
                               np.deg2rad(dlatgrid),
                               np.deg2rad(dlongrid),
                               np.deg2rad(180.),
                               prot, t0, rp/Rsun, rs/Rsun,
                               x[:,i], y[:,i])
     pbar.update(i)

# Apply visibility
fluxvtime = np.zeros((nfilt, nt))
for it in range(nt):
     for ifilt in range(nfilt):
          fluxvtime[ifilt,it] += np.sum(intfluxgrid[:,:,ifilt] * vis[it])

# Add the star (planet flux is already normalized, so just add 1)
sysflux = fluxvtime + 1

fnoised = np.zeros((nfilt, nt))
ferr    = np.zeros((nfilt, nt))
# Addnoise
for ifilt in range(nfilt):
     scnoise = noise[ifilt] / np.sqrt(necl)
     fnoised[ifilt] = sysflux[ifilt] + scnoise * np.random.randn(nt)
     ferr[ifilt]    =                  scnoise * np.ones(nt)

np.savetxt(os.path.join(outdir, 'time.txt'), t)
np.savetxt(os.path.join(outdir, 'flux.txt'), fnoised.T)
np.savetxt(os.path.join(outdir, 'ferr.txt'), ferr.T)
np.savetxt(os.path.join(outdir, 'wl.txt'), wlmid)

# Make plots of emission
fig, axes = plt.subplots(nrows=nfilt, ncols=1)
fig.set_size_inches(5,8)

vmax = np.max(intfluxgrid)
vmin = np.min(intfluxgrid)
extent = (-180, 180, -90, 90)
for i, ax in enumerate(axes):
     im = ax.imshow(np.roll(intfluxgrid[:,:,i], nlon // 2, axis=1),
                    origin='lower', vmin=vmin, vmax=vmax, extent=extent)
     plt.colorbar(im, ax=ax)

plt.savefig(os.path.join(outdir, 'fmaps.png'))
plt.close(fig)

# Make plots of temperature
# See Rauscher et al 2018 for explanation of the
# factor of pi. Following their labeling, intfluxgrid
# is:
#         (R_p / R_s)**2 * (M_p / M_s)
#
# so we need a factor of pi to be in the right units for
# eq. 8 used by fmap_to_tmap().
fig, axes = plt.subplots(nrows=nfilt, ncols=1)
fig.set_size_inches(5,8)

inttgrid = np.zeros(intfluxgrid.shape)
for i in range(nfilt):
     inttgrid[:,:,i] = utils.fmap_to_tmap(intfluxgrid[:,:,i] / np.pi,
                                          wlmid[i]*1e-6,
                                          rp, rs, ts, 0)
vmax = np.max(inttgrid)
vmin = np.min(inttgrid)
extent = (-180, 180, -90, 90)
for i, ax in enumerate(axes):
     im = ax.imshow(np.roll(inttgrid[:,:,i], nlon // 2, axis=1),
                    origin='lower', vmin=vmin, vmax=vmax, extent=extent)
     plt.colorbar(im, ax=ax)

plt.savefig(os.path.join(outdir, 'tmaps.png'))
plt.close(fig)

for i in range(nfilt):
     plt.scatter(t, sysflux[i], s=0.1)

plt.savefig(os.path.join(outdir, 'lightcurves.png'))
plt.close()

plt.figure(figsize=(3.5,5))
for i in range(nlat):
     for j in range(nlon):
          plt.semilogy(tgrid[:,i,j], p, color='gray')

for j in range(nlon):
     cmap = mpl.cm.hsv
     icol = lon[j] / 360.
     if icol < 0.0:
          icol += 1
          
     plt.semilogy(tgrid[:,nlat//2,j], p, color=cmap(icol),
                  label=lon[j])

plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (bars)')
plt.gca().invert_yaxis()
plt.tight_layout()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = inset_axes(plt.gca(), width='10%', height='25%',
                 loc='lower left')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=360))
cbar = plt.colorbar(sm, cax=cax, label=r'Longitude ($^\circ$)')
cbar.set_ticks(np.linspace(0, 360, 5, endpoint=True))

plt.savefig(os.path.join(outdir, 'tgrid.png'))
plt.close()

# Plot of transmission
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
fig.set_size_inches(16, 4)

wl = 10000. / wn
#plotlon = np.array([-180., -90., 0., 90.])
#plotlat = np.array([   0.,   0., 0.,  0.])
ilon = [0, nlon//4, nlon//2, 3*nlon//4]
ilat = [nlat//2, nlat//2, nlat//2, nlat//2]

# Determine vmax, vmin
vmax = -np.inf
vmin =  np.inf
for i in range(len(ilon)):
     trans = np.exp(-taugrid[ilat[i], ilon[i]])
     vmax = np.max((vmax, np.max(trans)))
     vmin = np.min((vmin, np.min(trans)))
     
for i in range(len(ilon)):
     ax = axes[i]
     logp = np.log10(p)
     maxlogp = np.max(logp)
     minlogp = np.min(logp)

     logwl = np.log10(wl)
     maxlogwl = np.max(logwl)
     minlogwl = np.min(logwl)

     tau = taugrid[ilat[i], ilon[i]]

     im = ax.imshow(np.flip(np.exp(-tau)), aspect='auto',
                    extent=(minlogwl, maxlogwl, maxlogp, minlogp),
                    cmap='magma', vmin=vmin, vmax=vmax)

     #ax.set_xlim((np.min(np.log10(filtwl)), np.max(np.log10(filtwl))))

     yticks = ax.get_yticks()
     ax.set_yticklabels([r"$10^{{{:.0f}}}$".format(y) for y in yticks])
     ax.set_ylim((maxlogp, minlogp))

     xticks = ax.get_xticks()
     ax.set_xticklabels(np.round(10.**xticks, 2))
     #ax.set_xlim((minlogwl, maxlogwl))

     ax.set_xlabel('Wavelength (um)')
     if i == 0:
          ax.set_ylabel('Pressure (bars)')
     ax.set_title('Lat: {}, Lon: {}'.format(np.round(lat[ilat[i]], 2),
                                            np.round(lon[ilon[i]], 2)))

     transform = mpl.transforms.blended_transform_factory(
          ax.transData, ax.transAxes)

     for j in range(nfilt):
          ax.plot(np.log10(filtwl[j]), 1.0 - filttrans[j]/10.,
                  transform=transform,
                  label='{:.2f} um'.format(wlmid[j]), linestyle='--')

     #ax.legend(frameon=False, fontsize=6)
     fig.colorbar(im, ax=ax)
     #plt.colorbar(label=r'$T_{above} - T_{layer}$')

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'tau.png'))
plt.close()
     
def blackbody(T, wn):
     nt  = len(T)
     nwn = len(wn)
     bb = np.zeros((nt, nwn))

     # Convert from /cm to /m
     wn_m = wn * 1e2
     for i in range(nt):
          for j in range(nwn):
               bb[i,j] = (2.0 * scc.h * scc.c**2 * wn_m[j]**3) \
                    * 1/(np.exp(scc.h * scc.c * wn_m[j] / scc.k / T[i]) - 1.0)

     return bb

cf = np.zeros((nlat, nlon, nlev, nwn))

for i in range(nlat):
     for j in range(nlon):
          bb = blackbody(tgrid[:,i,j], wn)
          trans = np.exp(-taugrid[i,j])
          dlp = np.zeros(nlev)
          dt = np.zeros((nlev, nwn))
          for k in range(nlev-1, -1, -1):
               if k == nlev-1:
                    dt[k,:] = 0.0
                    dlp[k]  = 0.0
                    cf[i,j,k,:] = 0.0
               else:
                    dt[k,:] = trans[k+1] - trans[k]
                    dlp[k] = np.log(rt.pressureProfile[k]) - \
                         np.log(rt.pressureProfile[k+1])
                    cf[i,j,k] = bb[k] * dt[k,:]/dlp[k]


# Plot of contribution functions (transmission, really)
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
fig.set_size_inches(16, 4)
cmap = mpl.cm.get_cmap('rainbow')
for i in range(len(axes)):
     ax = axes[i]
     for k in range(nwn):
          if k % 200 == 0:
               color = cmap(1 - k / nwn)
               label = '{} um'.format(np.round(10000./wn[k],2))
               alpha = 1
          else:
               color = 'gray'
               alpha = 0.01
               label = None
          ax.semilogy(cf[ilat[i],ilon[i],:,k], rt.pressureProfile/1e5,
                      color=color, alpha=alpha, label=label)

          ax.set_xlabel('Contribution')
          if i == 0:
               ax.set_ylabel('Pressure (bars)')
          ax.set_title('Lat: {}, Lon: {}'.format(
               np.round(lat[ilat[i]], 2),
               np.round(lon[ilon[i]], 2)))
          
plt.gca().invert_yaxis()
plt.legend(fontsize=6)
plt.savefig(os.path.join(outdir, 'cf.png'))

# Filter-integrated contribution functions
filter_cf = np.zeros((nlat, nlon, nlev, nfilt)) 
cf_trans = np.zeros((nlat, nlon, nlev, nwn)) # convolve cf with filt trans
for i in range(nfilt):
     #filt_max_wn = np.max(filtwn[i])
     #filt_min_wn = np.min(filtwn[i])
     interp = sci.interp1d(filtwn[i], filttrans[i], bounds_error=False,
                           fill_value=0.0)
     interptrans = interp(wn)
     integtrans  = np.trapz(interptrans)
     for j in range(nlat):
          print(j)
          for k in range(nlon):
              cf_trans[j, k, :, :] = \
                   cf[j, k, :, :] * interptrans
              # Integrate
              for l in range(nlev):
                  filter_cf[j, k, l, i] = \
                      np.trapz(cf_trans[j, k, l]) / \
                      integtrans
               
fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
fig.set_size_inches(16, 4)
cmap = mpl.cm.get_cmap('rainbow')
for i in range(len(axes)):
     ax = axes[i]
     for k in range(nfilt):
          color = cmap(1 - k / nfilt)
          label = os.path.split(filters[k])[1]
          alpha = 1

          ax.semilogy(filter_cf[ilat[i],ilon[i],:,k],
                      rt.pressureProfile/1e5, color=color,
                      alpha=alpha, label=label)

          ax.set_xlabel('Contribution')
          if i == 0:
               ax.set_ylabel('Pressure (bars)')
          ax.set_title('Lat: {}, Lon: {}'.format(
               np.round(lat[ilat[i]], 2),
               np.round(lon[ilon[i]], 2)))
          
plt.gca().invert_yaxis()
plt.legend(fontsize=6)
plt.savefig(os.path.join(outdir, 'cf-filters.png'))
plt.close()

# Locations of contribution function maxima
cf_max_maps = np.zeros((nlat, nlon, nfilt))

logp = np.log10(rt.pressureProfile/1e5)
order = np.argsort(logp)
for i in range(nlat):
    for j in range(nlon):
        for k in range(nfilt):
            spl = sci.UnivariateSpline(logp[order],
                                       filter_cf[i,j,order,k],
                                       k=4, s=0)
            roots = spl.derivative().roots()
            yroots = spl(roots)
            ypeak = np.max(yroots)
            xpeak = roots[np.argmax(yroots)]
            cf_max_maps[i,j,k] = xpeak

# Plot of location of maximum contribution
gridspec_kw = {}
gridspec_kw['width_ratios'] = np.concatenate((np.ones(nfilt), [0.1]))
fig, axes = plt.subplots(ncols=nfilt+1, gridspec_kw=gridspec_kw)
fig.set_size_inches(16,5)
vmin = np.min(cf_max_maps)
vmax = np.max(cf_max_maps)
extent = (-180, 180, -90, 90)
for i in range(nfilt):
    ax = axes[i]
    im = ax.imshow(np.roll(cf_max_maps[:,:,i], nlon//2, axis=1),
                   vmin=vmin, vmax=vmax, extent=extent, origin='lower')

fig.colorbar(im, cax=axes[-1], label='Log(p) (bars)', shrink=0.5)

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'cf-maxima.png'))
plt.close()

# Plot of contribution functions along the equator
gridspec_kw = {}
gridspec_kw['width_ratios'] = np.concatenate((np.ones(nfilt), [0.1]))
fig, axes = plt.subplots(ncols=nfilt+1, gridspec_kw=gridspec_kw)
fig.set_size_inches(16,5)
vmin = np.min(filter_cf)
vmax = np.max(filter_cf)
extent = (-180, 180, 2, -6)
for i in range(nfilt):     
     ax = axes[i]
     im = ax.imshow(np.roll(filter_cf[nlat//2,:,:,i].T, nlon//2,
                            axis=1), vmin=vmin, vmax=vmax, origin='lower',
                    extent=extent, aspect='auto')
     if i == 0:
         ax.set_ylabel('Log(p) (bars)')
     ax.set_xlabel('Longitude (deg)')

fig.colorbar(im, cax=axes[-1], label='Contribution', shrink=0.5)

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'cf-equator.png'))
plt.close()
            
# Some useful things
outdict = {}

outdict['fluxgrid'] = fluxgrid
outdict['taugrid'] = taugrid
outdict['wn'] = wn
outdict['tgrid'] = tgrid
outdict['vis'] = vis
outdict['inttgrid'] = inttgrid
outdict['intfluxgrid'] = intfluxgrid
outdict['filter_cf'] = filter_cf
outdict['p'] = rt.pressureProfile/1e5
outdict['lat'] = lat
outdict['lon'] = lon
outdict['cf'] = cf
outdict['filter_cf'] = filter_cf
outdict['cf_max_maps'] = cf_max_maps
np.save(os.path.join(outdir, 'output.npy'), outdict)
