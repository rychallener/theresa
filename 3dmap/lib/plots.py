import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def emaps(planet, eigeny, outdir, proj='ortho'):
    ncurves, ny = eigeny.shape

    if proj == 'ortho':
        extent = (-90, 90, -90, 90)
        fname = 'emaps-ecl.png'
    elif proj == 'rect':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-rect.png'
    elif proj == 'moll':
        extent = (-180, 180, -90, 90)
        fname = 'emaps-moll.png'

    lmax = np.int(ny**0.5 - 1)

    ncols = np.int(np.sqrt(ncurves) // 1)
    nrows = np.int(ncurves // ncols + (ncurves % ncols != 0))
    npane = ncols * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True)
    
    for j in range(ncurves):
        planet.map[1:,:] = 0

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]
        
        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1
        
        ax.imshow(planet.map.render(theta=0, projection=proj).eval(),
                  origin="lower",
                  cmap="plasma",
                  extent=extent)

        # Axes are wrong for non-rectangular projections
        if proj == 'ortho' or proj == 'moll':
            ax.axis('off')

    # Empty subplots
    for j in range(ncurves, npane):
        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]

        ax.axis('off')

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close(fig)

def lightcurves(t, lcs, outdir):
    nharm, nt = lcs.shape
    
    l =  1
    m = -1
    pos = True

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    for i in range(nharm):
        plt.plot(t, lcs[i], label=r"${}Y_{{{}{}}}$".format(["-", "+"][pos],
                                                           l, m))
        if pos:
            pos = False
        else:
            pos = True
            if l == m:
                l += 1
                m  = -l               
            else:
                m += 1
            
    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')
    plt.legend(ncol=l, fontsize=6)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'lightcurves.png'))
    plt.close()

def eigencurves(t, lcs, outdir, ncurves=None):
    if type(ncurves) == type(None):
        ncurves = lcs.shape[0]

    fig, ax = plt.subplots(1, figsize=(8,5))    

    for i in range(ncurves):
        plt.plot(t, lcs[i], label="E-curve {}".format(i+1))

    plt.ylabel('Normalized Flux')
    plt.xlabel('Time (days)')

    plt.legend(fontsize=6)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'eigencurves.png'))
    plt.close()
    
def ecurvepower(evalues, outdir):
    ncurves = len(evalues)
    num = np.arange(1, ncurves + 1)

    fig, axes = plt.subplots(nrows=2)
    
    axes[0].plot(num, evalues / np.sum(evalues), 'ob')
    axes[0].set_xlabel('E-curve Number')
    axes[0].set_ylabel('Normalized Power')

    axes[1].semilogy(num, evalues / np.sum(evalues), 'ob')
    axes[1].set_xlabel('E-curve Number')
    axes[1].set_ylabel('Normalized Power')

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'ecurvepower.png'))
    plt.close(fig)

def pltmaps(fit, proj='rect'):
    nmaps = len(fit.wlmid)

    ncols = np.int(np.sqrt(nmaps) // 1)
    nrows = nmaps // ncols + (nmaps % ncols != 0)

    xsize = 7. / 3. * ncols
    if proj == 'rect':
        ysize = 7. / 3. / 2. * nrows
    elif proj == 'ortho':
        ysize = 7. / 3. * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             sharey=True, squeeze=False)

    vmax = np.max([np.max(m.tmap[~np.isnan(m.tmap)]) for m in fit.maps])
    vmin = np.min([np.min(m.tmap[~np.isnan(m.tmap)]) for m in fit.maps])
    
    if proj == 'rect':
        extent = (-180, 180, -90, 90)
    elif proj == 'ortho':
        extent = (-90,   90, -90, 90)

    # The weird placement of the subplots in this figure is a long-
    # standing known bug in matplotlib with no straightforward
    # solution.  Probably not worth fixing here.  See
    # https://github.com/matplotlib/matplotlib/issues/5463
    for i in range(nmaps):
        irow = i // ncols
        icol = i %  ncols
        ax = axes[irow,icol]
        im = ax.imshow(fit.maps[i].tmap, origin='lower', cmap='plasma',
                       extent=extent, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Temperature (K)')
        ax.set_title('{:.2f} um'.format(fit.wlmid[i]))

    fig.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir,
                             'bestfit-{}-maps.png'.format(proj)))
    plt.close(fig)

def bestfit(fit):
    t = fit.t
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nfilt = len(fit.wlmid)
    nt = len(t)

    hratios = np.zeros(nfilt+1)
    hratios[0] = 0.5
    hratios[1:] = 0.5 / nfilt
    
    gridspec_kw = {'height_ratios':hratios}
    
    fig, axes = plt.subplots(nrows=nfilt+1, ncols=1, sharex=True,
                             gridspec_kw=gridspec_kw, figsize=(8,10))

    nt = len(t)
    
    for i in range(nfilt):
        axes[0].plot(t, fit.maps[i].bestfit, zorder=2, color=colors[i],
                     label='{:.2f} um'.format(fit.wlmid[i]))
        axes[0].scatter(t, fit.flux[i], s=0.1, zorder=1, color=colors[i])

    axes[0].legend()
    axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

    for i in range(nfilt):
        axes[i+1].scatter(t, fit.flux[i] - fit.maps[i].bestfit, s=0.1,
                          color=colors[i])
        axes[i+1].set_ylabel('Residuals')
        axes[i+1].axhline(0, 0, 1, color='black', linestyle='--')
        if i == nfilt-1:
            axes[i+1].set_xlabel('Time (days)')

    fig.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'bestfit-lcs.png'))
    plt.close(fig)

def ecurveweights(fit):
    nwl = len(fit.wlmid)
    ncurves = fit.cfg.twod.ncurves
    npar = ncurves + 2

    maxweight = -np.inf
    minweight =  np.inf

    if nwl == 1:
        shifts = [0]
    else:
        shifts = np.linspace(-0.2, 0.2, num=nwl, endpoint=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    for i in range(nwl):
        weights = fit.maps[i].bestp[:ncurves]
        uncs    = fit.maps[i].stdp[:ncurves]
        axes[0].errorbar(np.arange(ncurves) + shifts[i] + 1,
                         weights, uncs, fmt='o',
                         label="{:.2f} um".format(fit.wlmid[i]))
        axes[0].set_ylabel("E-curve weight")
        maxweight = np.max((maxweight, np.max(weights)))
        minweight = np.min((minweight, np.min(weights)))

        axes[1].scatter(np.arange(ncurves) + shifts[i] + 1,
                        np.abs(weights / uncs))
        axes[1].set_ylabel("E-curve Significance")
        axes[1].set_xlabel("E-curve number")
        axes[1].set_yscale('log')

    yrange = maxweight - minweight
    axes[0].set_ylim((minweight - 0.1 * yrange,
                      maxweight + 0.1 * yrange))
        
    axes[0].legend()

    xlim = axes[1].get_xlim()
    axes[1].hlines(3, 0, nwl*ncurves+1, linestyles='--', label=r'3$\sigma$')
    axes[1].set_xlim(xlim)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'ecurveweight.png'))
    plt.close()

def hshist(fit):
    '''
    Makes a plot of hotspot location posterior distribution
    '''
    nmaps = len(fit.maps)
    fig, axes = plt.subplots(nrows=2, ncols=nmaps, sharey='row')

    for i in range(nmaps):
        # Latitude
        ax = axes[0][i]
        ax.hist(fit.maps[i].hslocpost[0], bins=20)
        ax.set_xlabel('Latitude (deg)')
        ylim = ax.get_ylim()
        ax.vlines(fit.maps[i].hslocbest[0], ylim[0], ylim[1], color='red')
        ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel('Samples')
        # Longitude
        ax = axes[1][i]
        ax.hist(fit.maps[i].hslocpost[1], bins=20)
        ax.set_xlabel('Longitude (deg)')
        ylim = ax.get_ylim()
        ax.vlines(fit.maps[i].hslocbest[1], ylim[0], ylim[1], color='red')
        ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel('Samples')

    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'hotspot-hist.png'))
    plt.close()

def bestfitlcsspec(fit):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    nfilt, nt = fit.specbestmodel.shape

    hratios = np.zeros(nfilt+1)
    hratios[0] = 0.5
    hratios[1:] = 0.5 / nfilt
    
    gridspec_kw = {'height_ratios':hratios}
    
    fig, axes = plt.subplots(nrows=nfilt+1, ncols=1, sharex=True,
                             gridspec_kw=gridspec_kw, figsize=(8,10))

    for i in range(nfilt):
        axes[0].scatter(fit.t, fit.flux[i], s=0.1, zorder=1,
                        color=colors[i])
        axes[0].plot(fit.t, fit.specbestmodel[i],
                     label='{:.2f} um'.format(fit.wlmid[i]), zorder=2,
                     color=colors[i])

    axes[0].legend()
    axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

    for i in range(nfilt):
        axes[i+1].scatter(fit.t, fit.flux[i] - fit.specbestmodel[i], s=0.1,
                          color=colors[i])
        axes[i+1].set_ylabel('Residuals')
        axes[i+1].axhline(0, 0, 1, color='black', linestyle='--')
        if i == nfilt-1:
            axes[i+1].set_xlabel('Time (days)')

    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'bestfit-lcs-spec.png'))
    plt.close(fig)

def bestfittgrid(fit):
    fig, ax = plt.subplots(figsize=(6,8))

    # Match colors to light curves
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Line colors from colormap
    cmap = mpl.cm.get_cmap('hsv')

    nmaps = len(fit.maps)

    # Latitude index 
    ieq = fit.cfg.twod.nlat // 2

    for i in range(fit.cfg.twod.nlat):
        for j in range(fit.cfg.twod.nlon):
            lat = fit.lat[i,j]
            lon = fit.lon[i,j]
            if i == ieq:
                label = "Lat: {:.1f}, Lon: {:.1f}".format(lat, lon)
                
                ic = lon / 360.
                if ic < 0.0:
                    ic += 1.0
                    
                color = cmap(ic)
                zorder = 2
                alpha = 1.0
            else:
                label = None
                color = 'gray'
                zorder = 1
                alpha = 0.5
            
            if ((lon + fit.dlon < fit.minvislon) or
                (lon - fit.dlon > fit.maxvislon)):
                linestyle = '--'
            else:
                linestyle = '-'
                
            ax.semilogy(fit.besttgrid[:,i,j], fit.p, label=label,
                        linestyle=linestyle, color=color, zorder=zorder,
                        alpha=alpha)

            ax.scatter(fit.tmaps[:,i,j], fit.pmaps[:,i,j],
                       c=colors[:nmaps], marker='o', zorder=3, s=4)

    ax.invert_yaxis()
    ax.legend(ncol=2, fontsize=6)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bars)")
    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'bestfit-tp.png'))
    plt.close(fig)

def visanimation(fit, fps=60, step=10):
    fig = plt.figure()
    ims = []

    Writer = animation.writers['pillow']
    writer = Writer(fps=fps)

    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.yticks(np.linspace( -90,  90, 13, endpoint=True))
    plt.xticks(np.linspace(-180, 180, 13, endpoint=True))

    nt = len(fit.t)

    for i in range(0, nt, step):
        im = plt.imshow(fit.vis[i], animated=True,
                        vmax=np.max(fit.vis), vmin=np.min(fit.vis),
                        extent=(-180, 180, -90, 90))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50,
                                    blit=True, repeat_delay=1000)

    ani.save(os.path.join(fit.cfg.outdir, 'vis.gif'), dpi=300, writer=writer)

    plt.close(fig)

def fluxmapanimation(fit, fps=60, step=10):
    nmaps = len(fit.wlmid)

    ncols = np.min((nmaps, 3))
    nrows = nmaps // ncols + (nmaps % ncols != 0)

    xsize = 7. / 3. * ncols
    ysize = 7. / 3. / 2. * nrows

    figsize = (xsize, ysize)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             sharey=True, squeeze=False, figsize=figsize)

    vmax = np.max(fit.fmaps[~np.isnan(fit.fmaps)])
    vmin = np.min(fit.fmaps[~np.isnan(fit.fmaps)])
    
    extent = (-180, 180, -90, 90)

    all_ims   = []

    Writer = animation.writers['pillow']
    writer = Writer(fps=fps)
    
    for j in range(0, len(fit.t), step):
        frame_ims = []
        for i in range(nmaps):
            irow = i // ncols
            icol = i %  ncols
            ax = axes[irow,icol]
            im = ax.imshow(fit.fmaps[i]*fit.vis[j],
                           origin='lower', cmap='plasma',
                           extent=extent,
                           vmin=vmin, vmax=vmax)
            #plt.colorbar(im, ax=ax)
            ax.set_title('{:.2f} um'.format(fit.wlmid[i]))
            frame_ims.append(im)
            
        all_ims.append(frame_ims)
   
    ani = animation.ArtistAnimation(fig, all_ims, interval=50,
                                    blit=True, repeat_delay=1000)

    ani.save(os.path.join(fit.cfg.outdir, 'fmaps.gif'), dpi=300, writer=writer)

    plt.close(fig)


def tau(fit, ilat=None, ilon=None):
    fig, ax = plt.subplots()
    
    cfg = fit.cfg
    
    if type(ilat) == type(None):
        ilat = cfg.twod.nlat // 2
    if type(ilon) == type(None):
        ilon = cfg.twod.nlon // 2
        
    nlat, nlon = fit.taugrid.shape
    npress, nwn = fit.taugrid[0,0].shape
    wn = fit.modelwngrid
    wl = 10000 / fit.modelwngrid
    p = fit.p

    logp = np.log10(p)
    maxlogp = np.max(logp)
    minlogp = np.min(logp)

    logwl = np.log10(wl)
    maxlogwl = np.max(logwl)
    minlogwl = np.min(logwl)

    tau = fit.taugrid[ilat,ilon]
    
    plt.imshow(np.flip(tau), aspect='auto',
               extent=(minlogwl, maxlogwl, maxlogp, minlogp),
               cmap='magma')

    yticks = plt.yticks()[0]
    plt.yticks(yticks, [r"$10^{{{:.0f}}}$".format(y) for y in yticks])
    plt.ylim((maxlogp, minlogp))

    xticks = plt.xticks()[0]
    plt.xticks(xticks, np.round(10.**xticks, 2))
    plt.xlim((minlogwl, maxlogwl))

    plt.xlabel('Wavelength (um)')
    plt.ylabel('Pressure (bars)')

    nfilt = len(fit.filtwl)
    ax = plt.gca()
    transform = mpl.transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    # Note: assumes all filters are normalized to 1, and plots them
    # in the top tenth of the image.
    for i in range(nfilt):
        plt.plot(np.log10(fit.filtwl[i]), 1.0 - fit.filttrans[i]/10.,
                 transform=transform, label='{:.2f} um'.format(fit.wlmid[i]),
                 linestyle='--')

    plt.legend(frameon=False)
    plt.colorbar(label=r'$T_{above} - T_{layer}$')
    plt.savefig(os.path.join(fit.cfg.outdir, 'cf.png'))
    plt.close()

def pmaps3d(fit):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    nmaps = fit.pmaps.shape[0]

    tmax = np.max(fit.tmaps)
    tmin = np.min(fit.tmaps)
    for i in range(nmaps):
        cm = mpl.cm.coolwarm((fit.tmaps[i] - tmin)/(tmax - tmin))
        ax.plot_surface(fit.lat, fit.lon, fit.pmaps[i], facecolors=cm,
                        linewidth=5, shade=False)
        ax.plot_wireframe(fit.lat, fit.lon, fit.pmaps[i], linewidth=0.5,
                          color=colors[i])

    ax.invert_zaxis()
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Longitude (deg)')
    ax.set_zlabel('Pressure (bars)')
    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'pmaps.png'))
    plt.close()
