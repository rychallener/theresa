import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
import matplotlib.lines as mpll
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import atm


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
    plt.close(fig)

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
    plt.close(fig)
    
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

    naxes = nrows * ncols
    extra = nmaps % ncols

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
    for i in range(naxes):            
        irow = i // ncols
        icol = i %  ncols
        ax = axes[irow,icol]

        if i >= nmaps:
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False,
                           left=False, right=False)
            continue            
        
        im = ax.imshow(fit.maps[i].tmap, origin='lower', cmap='plasma',
                       extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title('{:.2f} um'.format(fit.wlmid[i]))

        if icol == 0:
            ax.set_ylabel(r'Latitude ($^\circ$)')
        if i >= naxes - ncols - (ncols - extra):
            ax.set_xlabel(r'Longitude ($^\circ$)')

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(im, cax=cax, label='Temperature (K)')
    plt.savefig(os.path.join(fit.cfg.twod.outdir,
                             'bestfit-{}-maps.png'.format(proj)))
    plt.close(fig)

def tmap_unc(fit, proj='rect'):
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

    naxes = nrows * ncols
    extra = nmaps % ncols

    vmax = np.max([np.max(m.tmapunc[~np.isnan(m.tmap)]) for m in fit.maps])
    vmin = np.min([np.min(m.tmapunc[~np.isnan(m.tmap)]) for m in fit.maps])
    
    if proj == 'rect':
        extent = (-180, 180, -90, 90)
    elif proj == 'ortho':
        extent = (-90,   90, -90, 90)

    # The weird placement of the subplots in this figure is a long-
    # standing known bug in matplotlib with no straightforward
    # solution.  Probably not worth fixing here.  See
    # https://github.com/matplotlib/matplotlib/issues/5463
    for i in range(naxes):            
        irow = i // ncols
        icol = i %  ncols
        ax = axes[irow,icol]

        if i >= nmaps:
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False,
                           left=False, right=False)
            continue            
        
        im = ax.imshow(fit.maps[i].tmapunc, origin='lower', cmap='plasma',
                       extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title('{:.2f} um'.format(fit.wlmid[i]))

        if icol == 0:
            ax.set_ylabel(r'Latitude ($^\circ$)')
        if i >= naxes - ncols - (ncols - extra):
            ax.set_xlabel(r'Longitude ($^\circ$)')

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(im, cax=cax, label='Temperature Uncertainty (K)')
    plt.savefig(os.path.join(fit.cfg.twod.outdir,
                             'bestfit-{}-maps-unc.png'.format(proj)))
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
        axes[0].plot(t, fit.maps[i].bestln.bestfit, zorder=2,
                     color=colors[i],
                     label='{:.2f} um'.format(fit.wlmid[i]))
        axes[0].scatter(t, fit.flux[i], s=0.1, zorder=1, color=colors[i])

    axes[0].legend()
    axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

    for i in range(nfilt):
        axes[i+1].scatter(t, fit.flux[i] - fit.maps[i].bestln.bestfit, s=0.1,
                          color=colors[i])
        axes[i+1].set_ylabel('Residuals')
        axes[i+1].axhline(0, 0, 1, color='black', linestyle='--')
        if i == nfilt-1:
            axes[i+1].set_xlabel('Time (days)')

    fig.tight_layout()
    plt.savefig(os.path.join(fit.cfg.twod.outdir, 'bestfit-lcs.png'))
    plt.close(fig)

def ecurveweights(fit):
    nwl = len(fit.wlmid)

    maxweight = -np.inf
    minweight =  np.inf

    maxcurves = np.max([m.bestln.ncurves for m in fit.maps])

    if nwl == 1:
        shifts = [0]
    else:
        shifts = np.linspace(-0.2, 0.2, num=nwl, endpoint=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    for i in range(nwl):
        ncurves = fit.maps[i].bestln.ncurves
        npar = ncurves + 2
        weights = fit.maps[i].bestln.bestp[:ncurves]
        uncs    = fit.maps[i].bestln.stdp[:ncurves]
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
    axes[1].hlines(3, 0, nwl*maxcurves+1, linestyles='--',
                   label=r'3$\sigma$')
    axes[1].set_xlim(xlim)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.twod.outdir, 'ecurveweight.png'))
    plt.close(fig)

def hshist(fit):
    '''
    Makes a plot of hotspot location posterior distribution
    '''
    nmaps = len(fit.maps)
    fig, axes = plt.subplots(nrows=2, ncols=nmaps, sharey='row',
                             squeeze=False)

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
    plt.savefig(os.path.join(fit.cfg.twod.outdir, 'hotspot-hist.png'))
    plt.close(fig)

def bestfitlcsspec(fit, outdir=''):
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
    plt.savefig(os.path.join(outdir, 'bestfit-lcs-spec.png'))
    plt.close(fig)

def bestfittgrid(fit, outdir=''):
    fig, ax = plt.subplots(figsize=(6,8))

    # Match colors to light curves
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Line colors from colormap
    cmap = mpl.cm.get_cmap('hsv')

    nmaps = len(fit.maps)

    # Latitude index 
    ieq = fit.cfg.twod.nlat // 2

    cfnorm_lines = np.nanmax(fit.cf)
    cfnorm_dots  = np.nanmax(np.sum(fit.cf, axis=2))
    
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
            else:
                label = None
                color = 'gray'
                zorder = 1
            
            if ((lon + fit.dlon < fit.minvislon) or
                (lon - fit.dlon > fit.maxvislon)):
                linestyle = '--'
            else:
                linestyle = '-'

            points = np.array([fit.besttgrid[:,i,j], fit.p]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]],
                                      axis=1)
            norm = plt.Normalize(0, 1)

            lc = collections.LineCollection(segments,
                                            cmap=gradient_cmap(color),
                                            norm=norm, zorder=zorder)
            lc.set_array(np.max(fit.cf[i,j,:-1], axis=1) / cfnorm_lines)
            line = ax.add_collection(lc)

            if linestyle != '--':
                for k in range(nmaps):
                    alpha = np.sum(fit.cf[i,j,:,k]) / cfnorm_dots
                    alpha = np.round(alpha, 2)
                    ax.scatter(fit.tmaps[k,i,j], fit.pmaps[k,i,j],
                               c=colors[k], marker='o', zorder=3, s=1,
                               alpha=alpha)

    # Build custom legend
    legend_elements = []
    for i in range(nmaps):
        label = str(np.round(fit.wlmid[i], 2)) + ' um'
        legend_elements.append(mpll.Line2D([0], [0], color='w',
                                           label=label,
                                           marker='o',
                                           markerfacecolor=colors[i],
                                           markersize=4))

    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.legend(handles=legend_elements, loc='best')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bars)")
    plt.tight_layout()

    cax = inset_axes(plt.gca(), width='5%', height='25%',
                     loc='lower right')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=360))
    cbar = plt.colorbar(sm, cax=cax, label=r'Longitude ($^\circ$)')
    cbar.set_ticks(np.linspace(0, 360, 5, endpoint=True))
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    
    plt.savefig(os.path.join(outdir, 'bestfit-tp.png'))
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


def tau(fit, ilat=None, ilon=None, outdir=''):
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
    
    plt.imshow(np.flip(np.exp(-tau)), aspect='auto',
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

    leg = plt.legend(frameon=False, ncol=4, fontsize=8)
    for text in leg.get_texts():
        text.set_color("white")
        
    plt.colorbar(label=r'$e^{-\tau}$')
    plt.savefig(os.path.join(outdir, 'transmission.png'))
    plt.close(fig)

def pmaps3d(fit, animate=False, outdir=''):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    nmaps = fit.pmaps.shape[0]

    tmax = np.nanmax(fit.tmaps)
    tmin = np.nanmin(fit.tmaps)

    def init():
        for i in range(nmaps):
            cm = mpl.cm.coolwarm((fit.tmaps[i] - tmin)/(tmax - tmin))
            ax.plot_surface(fit.lat, fit.lon, np.log10(fit.pmaps[i]),
                            facecolors=cm, linewidth=3, shade=False)
            ax.plot_wireframe(fit.lat, fit.lon,
                              np.log10(fit.pmaps[i]), linewidth=0.5,
                              color=colors[i])

        ax.invert_zaxis()
        ax.set_xlabel('Latitude (deg)')
        ax.set_ylabel('Longitude (deg)')
        ax.set_zlabel('log(p) (bars)')
        plt.tight_layout()
        return fig,

    init()
    plt.savefig(os.path.join(fit.cfg.outdir, 'pmaps.png'))
    plt.close(fig)
    
    if not animate:
        return

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    nframes = 80
    
    Writer = animation.writers['pillow']
    writer = Writer(fps=15)

    base_azim = 45.0
    base_elev = 15.0

    azim_vary = np.concatenate((np.linspace(0., 45., nframes // 4),
                                np.linspace(45., 0., nframes // 4),
                                np.zeros(nframes // 2)))
    azim = base_azim + azim_vary

    elev_vary = np.concatenate((np.zeros(nframes // 2),
                                np.linspace(0., 30., nframes // 4),
                                np.linspace(30., 0., nframes // 4)))
    elev = base_elev + elev_vary
    
    def animate(i):
        ax.view_init(elev=elev[i], azim=azim[i])
        return fig,
        
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nframes, interval=20, blit=True)

    anim.save(os.path.join(outdir, 'pmaps3d.gif'), dpi=300,
              writer=writer)

    plt.close(fig)
    
def tgrid_unc(fit, outdir=''):
    '''
    Plots the temperature profiles of the atmosphere at various
    important locations, with uncertainties.
    '''
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True,
                             sharey=True)

    mcmcout = np.load(fit.cfg.outdir + '/3dmcmc.npz')

    niter, npar = fit.posterior3d.shape
    nlev, nlat, nlon = fit.besttgrid.shape

    # Limit calculations if large number of samples
    ncalc = np.min((5000, niter))
    
    tgridpost = np.zeros((ncalc, nlev, nlat, nlon))
    for i in range(ncalc):
        ipost = i * niter // ncalc
        pmaps = atm.pmaps(fit.posterior3d[ipost], fit)
        tgridpost[i], p = atm.tgrid(nlev, nlat, nlon, fit.tmaps,
                                    pmaps, fit.cfg.threed.pbot,
                                    fit.cfg.threed.ptop,
                                    fit.posterior3d[ipost],
                                    fit.nparams3d, fit.modeltype3d,
                                    fit.imodel3d,
                                    interptype=fit.cfg.threed.interp,
                                    smooth=fit.cfg.threed.smooth)

    # Collapse to 1D for easier indexing
    lat = np.unique(fit.lat)
    lon = np.unique(fit.lon)
    
    for i in range(ncols*nrows):
        irow = i // nrows
        icol = i %  ncols
        ax = axes[irow, icol]
        # Hotspot
        if i == 0:
            # Average over all maps
            hslatavg = np.mean([a.hslocbest[0] for a in fit.maps])
            hslonavg = np.mean([a.hslocbest[1] for a in fit.maps])
            ilat = np.abs(lat - hslatavg).argmin()
            ilon = np.abs(lon - hslonavg).argmin()
            title = 'Hotspot'
        # Substellar point
        if i == 1:
            ilat = np.abs(lat -  0.0).argmin()
            ilon = np.abs(lon -  0.0).argmin()
            title = 'Substellar'
        # West terminator
        if i == 2:
            ilat = np.abs(lat -  0.0).argmin()
            ilon = np.abs(lon + 90.0).argmin()
            title = 'West Terminator'
        # East terminator
        if i == 3:
            ilat = np.abs(lat -  0.0).argmin()
            ilon = np.abs(lon - 90.0).argmin()
            title = 'East Terminator'

        tdist = tgridpost[:,:,ilat,ilon]

        l1 = np.percentile(tdist, 15.87, axis=0)
        l2 = np.percentile(tdist,  2.28, axis=0)
        h1 = np.percentile(tdist, 84.13, axis=0)
        h2 = np.percentile(tdist, 97.72, axis=0)

        bf = fit.besttgrid[:,ilat,ilon]

        ax.fill_betweenx(fit.p, l2, h2, facecolor='royalblue')
        ax.fill_betweenx(fit.p, l1, h1, facecolor='cornflowerblue')
        ax.semilogy(bf, fit.p, label='Best Fit', color='black')
        if irow == 1:
            ax.set_xlabel('Temperature (K)')
        if icol == 0:
            ax.set_ylabel('Pressure (bars)')
        if i == 0:
            plt.gca().invert_yaxis()

        subtitle = r'$\theta={}, \phi={}$'.format(lat[ilat], lon[ilon])
        ax.set_title(title + '\n' + subtitle)


    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'tgrid_unc.png'))
    plt.close(fig)

def cf_by_location(fit, outdir=''):
    nlat, nlon, nlev, nfilt = fit.cf.shape
    fig, axes = plt.subplots(nrows=nlat, ncols=nlon, sharey=True, sharex=True)
    fig.set_size_inches(16, 8)

    # Place labels on a single large axes object
    bigax = fig.add_subplot(111, frameon=False)
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor='w', top=False, bottom=False,
                      left=False, right=False)

    bigax.set_ylabel('Pressure (bars)', labelpad=20)
    bigax.set_xlabel('Contribution (arbitrary)', labelpad=10)

    
    cmap = mpl.cm.get_cmap('rainbow')
    for i in range(nlat):
        for j in range(nlon):
            ax = axes[i,j]
            for k in range(nfilt):
                color = cmap(k / nfilt)
                label = os.path.split(fit.cfg.twod.filtfiles[k])[1]

                ax.semilogy(fit.cf[i,j,:,k], fit.p, color=color,
                            label=label)

            if i == nlat - 1:
                ax.set_xlabel(r'{}$^\circ$'.format(np.round(fit.lon[i,j], 2)))
            if j == 0:
                ax.set_ylabel(r'{}$^\circ$'.format(np.round(fit.lat[i,j], 2)))
            if i == nlat -1 and j == nlon - 1:
                ax.invert_yaxis()

            ax.set_xticklabels([])
            ax.tick_params(axis='y', labelsize=6)

    # Since we share y axes, this inverts them all
    #plt.gca().invert_yaxis()
    #plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cf.png'))
    plt.close(fig)

def cf_by_filter(fit, outdir=''):
    nlat, nlon, nlev, nfilt = fit.cf.shape
    
    ncols = np.int(np.sqrt(nfilt) // 1)
    nrows = np.int((nfilt // ncols) + (nfilt % ncols != 0))
    naxes = nrows * ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             sharey=True)
    fig.set_size_inches(8, 8)

    extra = nfilt % ncols

    ieq = nlat // 2

    for i in range(naxes):
        irow = i // ncols
        icol = i %  ncols

        ax = axes[irow, icol]

        # Hide extra axes and move on
        if i >= nfilt:
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False,
                           left=False, right=False)
            continue

        cmap = mpl.cm.get_cmap('hsv')
        
        for j in range(nlat):
            for k in range(nlon):
                if j == ieq:
                    ic = fit.lon[j,k] / 360.
                    if ic < 0:
                        ic += 1

                    color = cmap(ic)
                    label = r"${} ^\circ$".format(np.round(fit.lon[j,k], 2))
                    zorder = 1
                else:
                    color = 'gray'
                    label = None
                    zorder = 0
                    
                ax.semilogy(fit.cf[j,k,:,i], fit.p, color=color,
                            label=label, zorder=zorder)

        if icol == 0:
            ax.set_ylabel('Pressure (bars)')
        if i >= naxes - ncols - (ncols - extra):
            ax.set_xlabel('Contribution (arbitrary)')

        ax.set_title("{} um".format(np.round(fit.wlmid[i], 2)))

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cf-by-filter.png'))
    plt.close(fig)

        
def cf_slice(fit, ilat=None, ilon=None, fname=None, outdir=''):   
    if ilat is not None and ilon is not None:
        print("Must specify either ilat or ilon, not both.")
        return

    nlat, nlon, nlev, nfilt = fit.cf.shape 
    logp = np.log10(fit.p)
    minlogp = np.min(logp)
    maxlogp = np.max(logp)
    
    # Default behavior is slice along the equator
    if   ilat is     None and ilon is     None:
        latslice = nlat // 2
        lonslice = np.arange(nlon)
        xmin = -180.
        xmax =  180.
        xlabel = 'Longitude (deg)'
    elif ilat is     None and ilon is not None:
        latslice = np.arange(nlat)
        lonslice = ilon
        xmin = -90.
        xmax =  90.
        xlabel = 'Latitude (deg)'
    elif ilat is not None and ilon is     None:
        latslice = ilat
        lonslice = np.arange(nlon)
        xmin = -180.
        xmax =  180.
        xlabel = 'Longitude (deg)'

    if fname is None:
        fname = 'cf-slice.png'

    gridspec_kw = {}
    gridspec_kw['width_ratios'] = np.concatenate((np.ones(nfilt), [0.1]))

    fig, axes = plt.subplots(ncols=nfilt + 1, gridspec_kw=gridspec_kw)
    fig.set_size_inches(3*nfilt+1, 5)

    vmin = np.nanmin(fit.cf[latslice, lonslice])
    vmax = np.nanmax(fit.cf[latslice, lonslice])

    extent = (xmin, xmax, maxlogp, minlogp)
    
    for i in range(nfilt):
        ax = axes[i]
        im = ax.imshow(fit.cf[latslice, lonslice,:,i].T, vmin=vmin,
                       vmax=vmax, origin='lower', extent=extent,
                       aspect='auto')

        if ilon is None:
            ax.plot(fit.lon[latslice],
                    np.log10(fit.pmaps[i,latslice,lonslice]), color='red')
        else:
            ax.plot(fit.lat[:,lonslice],
                    np.log10(fit.pmaps[i,latslic,lonslice]), color='red')
            
        if i == 0:
            ax.set_ylabel('Log(p) (bars)')

        ax.set_xlabel(xlabel)

    fig.colorbar(im, cax=axes[-1], label='Contribution')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close(fig)
        
# Function adapted from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def gradient_cmap(color):
    '''
    Utility function to make colormaps which are a 
    gradient from white to the specified color.
    '''
    rgb_color = mplc.to_rgb(color)
    dec_color = np.array(rgb_color) #/ 256

    white = [1., 1., 1.]

    dec_colors = [white, dec_color]

    cdict = {}

    # Just two colors
    loclist = [0,1]

    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[loclist[i], dec_colors[i][num], dec_colors[i][num]] \
                    for i in range(2)]
        cdict[col] = col_list

    cmap = mplc.LinearSegmentedColormap(color, segmentdata=cdict, N=256)
    return cmap
