import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
import matplotlib.lines as mpll
import matplotlib.colors as mplc
import matplotlib.ticker as mplt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import atm
import utils
import copy


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
    nmaps = fit.nmaps

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

    vmax = np.max(fit.tmaps2d[~np.isnan(fit.tmaps2d)])
    vmin = np.min(fit.tmaps2d[~np.isnan(fit.tmaps2d)])
    
    if proj == 'rect':
        extent = (-180, 180, -90, 90)
    elif proj == 'ortho':
        extent = (-90,   90, -90, 90)

    # The weird placement of the subplots in this figure is a long-
    # standing known bug in matplotlib with no straightforward
    # solution.  Probably not worth fixing here.  See
    # https://github.com/matplotlib/matplotlib/issues/5463
    imap = 0
    for d in fit.datasets:
        for m in d.maps:           
            irow = imap // ncols
            icol = imap %  ncols
            ax = axes[irow,icol]

            if imap >= nmaps:
                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.tick_params(labelcolor='w', top=False, bottom=False,
                               left=False, right=False)
                continue            

            im = ax.imshow(m.tmap, origin='lower', cmap='plasma',
                           extent=extent, vmin=vmin, vmax=vmax)

            ax.set_title('{:.2f} um'.format(m.wlmid))

            if icol == 0:
                ax.set_ylabel(r'Latitude ($^\circ$)')
            if imap >= naxes - ncols - (ncols - extra):
                ax.set_xlabel(r'Longitude ($^\circ$)')

            imap += 1

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(im, cax=cax, label='Temperature (K)')
    plt.savefig(os.path.join(fit.cfg.twod.outdir,
                             'bestfit-{}-maps.png'.format(proj)))
    plt.close(fig)

def tmap_unc(fit, proj='rect'):
    nmaps = fit.nmaps

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

    vmin =  np.inf
    vmax = -np.inf
    for d in fit.datasets:
        for m in d.maps:
            vmin = np.min((vmin, np.nanmin(m.tmapunc)))
            vmax = np.max((vmax, np.nanmax(m.tmapunc)))
    
    if proj == 'rect':
        extent = (-180, 180, -90, 90)
    elif proj == 'ortho':
        extent = (-90,   90, -90, 90)

    # The weird placement of the subplots in this figure is a long-
    # standing known bug in matplotlib with no straightforward
    # solution.  Probably not worth fixing here.  See
    # https://github.com/matplotlib/matplotlib/issues/5463
    imap = 0
    for d in fit.datasets:
        for m in d.maps:         
            irow = imap // ncols
            icol = imap %  ncols
            ax = axes[irow,icol]

            if imap >= nmaps:
                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.tick_params(labelcolor='w', top=False, bottom=False,
                               left=False, right=False)
                continue            

            im = ax.imshow(m.tmapunc, origin='lower', cmap='plasma',
                           extent=extent, vmin=vmin, vmax=vmax)

            ax.set_title('{:.2f} um'.format(m.wlmid))

            if icol == 0:
                ax.set_ylabel(r'Latitude ($^\circ$)')
            if imap >= naxes - ncols - (ncols - extra):
                ax.set_xlabel(r'Longitude ($^\circ$)')

            imap += 1

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(im, cax=cax, label='Temperature Uncertainty (K)')
    plt.savefig(os.path.join(fit.cfg.twod.outdir,
                             'bestfit-{}-maps-unc.png'.format(proj)))
    plt.close(fig)
    
def bestfit(fit, outdir=''):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for d in fit.datasets:
        nmaps = len(d.maps)
        
        hratios = np.zeros(nmaps+1)
        hratios[0] = 0.5
        hratios[1:] = 0.5 / nmaps
    
        gridspec_kw = {'height_ratios':hratios}
        for v in d.visits:
            fig, axes = plt.subplots(nrows=nmaps+1, ncols=1,
                                     sharex=True,
                                     gridspec_kw=gridspec_kw,
                                     figsize=(8,10))
            
            t = v.t - fit.cfg.planet.t0
            
            imap = 0
            for m in d.maps:
                where = np.where((d.t >= v.t[0] ) &
                                 (d.t <= v.t[-1]))
                axes[0].plot(t, m.bestln.bestfit[where], zorder=2,
                             color=colors[imap],
                             label='{:.2f} um'.format(m.wlmid))
                axes[0].scatter(t, m.flux[where], s=0.1, zorder=1,
                                color=colors[imap])
                imap += 1

            axes[0].legend()
            axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

            imap = 0
            for m in d.maps:
                where = np.where((d.t >= v.t[0] ) &
                                 (d.t <= v.t[-1]))
                res = m.flux[where] - m.bestln.bestfit[where]
                axes[imap+1].scatter(t, res, s=0.1,
                                     color=colors[imap])
                axes[imap+1].set_ylabel('Residuals')
                axes[imap+1].axhline(0, 0, 1, color='black', linestyle='--')
                if imap == nmaps-1:
                    axes[imap+1].set_xlabel('Time (days from transit)')
                imap += 1

            fig.tight_layout()
            fname = os.path.join(outdir,
                                 'bestfit-lcs-{}-{}.png'.format(d.name,
                                                                v.name))
                                                                     
            plt.savefig(fname)
            plt.close(fig)

def ecurveweights(fit):
    nmaps = fit.nmaps

    maxweight = -np.inf
    minweight =  np.inf

    maxcurves = -np.inf
    for d in fit.datasets:
        for m in d.maps:
            maxcurves = np.max((maxcurves, m.bestln.ncurves))

    if nmaps == 1:
        shifts = [0]
    else:
        shifts = np.linspace(-0.2, 0.2, num=nmaps, endpoint=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    i = 0
    for d in fit.datasets:
        for m in d.maps:
            ncurves = m.bestln.ncurves
            # No weights to plot
            if ncurves == 0:
                continue
            npar = ncurves + 2
            weights = m.bestln.bestp[:ncurves]
            uncs    = m.bestln.stdp[:ncurves]
            axes[0].errorbar(np.arange(ncurves) + shifts[i] + 1,
                             weights, uncs, fmt='o',
                             label="{:.2f} um".format(m.wlmid))
            axes[0].set_ylabel("E-curve weight")
            maxweight = np.max((maxweight, np.max(weights)))
            minweight = np.min((minweight, np.min(weights)))

            axes[1].scatter(np.arange(ncurves) + shifts[i] + 1,
                            np.abs(weights / uncs))
            axes[1].set_ylabel("E-curve Significance")
            axes[1].set_xlabel("E-curve number")
            axes[1].set_yscale('log')
            i += 1

    # In case every map was fit with a uniform model
    # (This plot is useless in that case, but at least we prevent crashes)
    if minweight == np.inf:
        minweight = 0.0
    if maxweight == -np.inf:
        maxweight = 1.0

    yrange = maxweight - minweight
    axes[0].set_ylim((minweight - 0.1 * yrange,
                      maxweight + 0.1 * yrange))
        
    axes[0].legend()

    xlim = axes[1].get_xlim()
    axes[1].hlines(3, 0, nmaps*maxcurves+1, linestyles='--',
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
    nmaps = fit.nmaps
    fig, axes = plt.subplots(nrows=2, ncols=nmaps, sharey='row',
                             squeeze=False)

    i = 0
    for d in fit.datasets:
        for m in d.maps:
            # Latitude
            ax = axes[0][i]
            ax.hist(m.hslocpost[0], bins=20)
            ax.set_xlabel('Latitude (deg)')
            ylim = ax.get_ylim()
            ax.vlines(m.hslocbest[0], ylim[0], ylim[1], color='red')
            ax.set_ylim(ylim)
            if i == 0:
                ax.set_ylabel('Samples')
            # Longitude
            ax = axes[1][i]
            ax.hist(m.hslocpost[1], bins=20)
            ax.set_xlabel('Longitude (deg)')
            ylim = ax.get_ylim()
            ax.vlines(m.hslocbest[1], ylim[0], ylim[1], color='red')
            ax.set_ylim(ylim)
            if i == 0:
                ax.set_ylabel('Samples')
            i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.twod.outdir, 'hotspot-hist.png'))
    plt.close(fig)

def bics(fit, outdir=''):
    nmaps = fit.nmaps
    
    fig, axes = plt.subplots(nrows=1, ncols=nmaps, squeeze=False)

    im = 0
    for d in fit.datasets:
        for m in d.maps:
            lmax    = fit.cfg.twod.lmax
            ncurves = fit.cfg.twod.ncurves

            ls = np.arange(1, lmax + 1)
            ns = np.arange(0, ncurves + 1)

            bicarray = np.zeros((lmax, ncurves+1))
            for il, l in enumerate(ls):
                for ic, n in enumerate(ns):
                    if hasattr(m, 'l{}n{}'.format(l,n)):
                        bicarray[il,ic] = getattr(
                            m, 'l{}n{}'.format(l,n)).bic
                    else:
                        bicarray[il,ic] = np.inf

            ax = axes[0,im]

            cmap = copy.copy(mpl.cm.get_cmap('viridis'))
            cmap.set_bad(color='red')
            overlaycmap = mplc.ListedColormap([(0,0,0,0), (0,0,0,1)])

            dbic = bicarray - np.nanmin(bicarray[bicarray != np.inf])

            extent = (-0.5, ncurves + 0.5, 0.5, lmax + 0.5)
            image = ax.imshow(dbic, interpolation='none', origin='lower',
                              cmap=cmap, norm=mplc.LogNorm(vmin=1, vmax=100),
                              extent=extent)
            # Super janky way to handle the infs (cover them with black squares)
            ax.imshow(~np.isfinite(dbic), interpolation='none', origin='lower',
                      cmap=overlaycmap, extent=extent)

            ax.set_xlabel('Number of Eigencurves')
            ax.set_ylabel(r'$l_{\rm max}$')
            plt.colorbar(image, ax=ax, label=r'$\Delta {\rm BIC}$',
                         extend='both')
            im += 1
        
    plt.savefig(os.path.join(outdir, 'bics.png'))
    plt.close(fig)

def bestfitlcsspec(fit, outdir=''):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    nmaps = fit.nmaps

    hratios = np.zeros(nmaps+1)
    hratios[0] = 0.5
    hratios[1:] = 0.5 / nmaps
    
    gridspec_kw = {'height_ratios':hratios}
    
    fig, axes = plt.subplots(nrows=nmaps+1, ncols=1, sharex=True,
                             gridspec_kw=gridspec_kw, figsize=(8,10))

    i = 0
    for d in fit.datasets:
        for m in d.maps:
            axes[0].scatter(d.t, m.flux, s=0.1, zorder=1,
                            color=colors[i])
            axes[0].plot(d.t, fit.specbestmodel[i],
                         label='{:.2f} um'.format(m.wlmid), zorder=2,
                         color=colors[i])
            i += 1

    axes[0].legend()
    axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

    i = 0
    for d in fit.datasets:
        for m in d.maps:
            axes[i+1].scatter(d.t, m.flux - fit.specbestmodel[i],
                              s=0.1,
                              color=colors[i])
            axes[i+1].set_ylabel('Residuals')
            axes[i+1].axhline(0, 0, 1, color='black', linestyle='--')
            if i == nmaps-1:
                axes[i+1].set_xlabel('Time (days)')
            i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-lcs-spec.png'))
    plt.close(fig)

def bestfittgrid(fit, outdir=''):
    fig, ax = plt.subplots(figsize=(6,8))

    # Match colors to light curves
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Line colors from colormap
    cmap = mpl.cm.get_cmap('hsv')

    nmaps = fit.nmaps

    cfnorm_lines = np.nanmax(fit.cf)
    cfnorm_dots  = np.nanmax(np.sum(fit.cf, axis=1))
    
    for i in range(fit.ncolumn):
        lat = fit.lat3d[i]
        lon = fit.lon3d[i]

        label = "Lat: {:.1f}, Lon: {:.1f}".format(lat, lon)

        ic = lon / 360.
        if ic < 0.0:
            ic += 1.0

        color = cmap(ic)
        zorder = 2

        minvislon = np.min([d.minvislon for d in fit.datasets])
        maxvislon = np.max([d.maxvislon for d in fit.datasets])

        if i not in fit.ivis3d:
            linestyle = '--'
        else:
            linestyle = '-'

        points = np.array([fit.besttgrid[:,i], fit.p]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]],
                                  axis=1)
        norm = plt.Normalize(0, 1)

        # Set up CF shading
        alpha = np.max(fit.cf[i,:-1], axis=1) / cfnorm_lines
        rgba = np.zeros((len(segments), 4))
        rgba[:,:3] = color[:3]
        rgba[:,3]  = alpha

        lc = collections.LineCollection(segments,
                                        colors=rgba,
                                        norm=norm,
                                        zorder=zorder)

        line = ax.add_collection(lc)

        if linestyle != '--':
            for k in range(nmaps):
                alpha = np.sum(fit.cf[i,:,k]) / cfnorm_dots
                alpha = np.round(alpha, 2)
                ax.scatter(fit.tmaps3d[k,i], fit.pmaps[k,i],
                           c=colors[k], marker='o', zorder=3, s=1,
                           alpha=alpha)

    # Build custom legend
    legend_elements = []
    i = 0
    for d in fit.datasets:
        for m in d.maps:
            label = str(np.round(m.wlmid, 2)) + ' um'
            legend_elements.append(mpll.Line2D([0], [0], color='w',
                                               label=label,
                                               marker='o',
                                               markerfacecolor=colors[i],
                                               markersize=4))
            i += 1

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

def visanimation(fit, fps=60, step=10, outdir=''):
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

    ani.save(os.path.join(outdir, 'vis.gif'), dpi=300, writer=writer)

    plt.close(fig)

def fluxmapanimation(fit, fps=60, step=10, outdir=''):
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

    ani.save(os.path.join(outdir, 'fmaps.gif'), dpi=300, writer=writer)

    plt.close(fig)


def tau(fit, icolumn=None, outdir=''):
    fig, ax = plt.subplots()
    
    cfg = fit.cfg

    # Default to column nearest the substellar point
    if type(icolumn) == type(None):
        dist = (fit.lat3d**2 + fit.lon3d**2)**0.5
        icolumn = dist.argmin()
        
    ncolumn = fit.taugrid.shape
    npress, nwn = fit.taugrid[0].shape
    wn = fit.modelwngrid
    wl = 10000 / fit.modelwngrid
    p = fit.p

    logp = np.log10(p)
    maxlogp = np.max(logp)
    minlogp = np.min(logp)

    logwl = np.log10(wl)
    maxlogwl = np.max(logwl)
    minlogwl = np.min(logwl)

    tau = fit.taugrid[icolumn]
    
    plt.imshow(np.flip(np.exp(-tau)), aspect='auto',
               extent=(minlogwl, maxlogwl, maxlogp, minlogp),
               cmap='magma', vmax=1.0, vmin=0.0)

    yticks = plt.yticks()[0]
    plt.yticks(yticks, [r"$10^{{{:.0f}}}$".format(y) for y in yticks])
    plt.ylim((maxlogp, minlogp))

    xticks = plt.xticks()[0]
    plt.xticks(xticks, np.round(10.**xticks, 2))
    plt.xlim((minlogwl, maxlogwl))

    plt.xlabel('Wavelength (um)')
    plt.ylabel('Pressure (bars)')

    nmaps = fit.nmaps
    ax = plt.gca()
    transform = mpl.transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    # Note: assumes all filters are normalized to 1, and plots them
    # in the top tenth of the image.
    for d in fit.datasets:
        for m in d.maps:
            plt.plot(np.log10(m.filtwl),
                     1.0 - m.filttrans/10.,
                     transform=transform,
                     label='{:.2f} um'.format(m.wlmid),
                     linestyle='--')

    leg = plt.legend(frameon=False, ncol=4, fontsize=8)
    for text in leg.get_texts():
        text.set_color("white")
        
    plt.colorbar(label=r'$e^{-\tau}$')
    plt.savefig(os.path.join(outdir, 'transmission.png'))
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

    mcmcout = np.load(outdir + '/3dmcmc.npz')

    niter, npar = fit.posterior3d.shape
    nlev, ncolumn = fit.besttgrid.shape

    # Limit calculations if large number of samples
    ncalc = np.min((5000, niter))
    
    tgridpost = np.zeros((ncalc, nlev, fit.ncolumn))
    for i in range(ncalc):
        ipost = i * niter // ncalc
        pmaps = atm.pmaps(fit.posterior3d[ipost], fit)
        tgridpost[i], p = atm.tgrid(nlev, fit.ncolumn, fit.tmaps3d,
                                    pmaps, fit.cfg.threed.pbot,
                                    fit.cfg.threed.ptop,
                                    fit.posterior3d[ipost],
                                    fit.nparams3d, fit.modeltype3d,
                                    fit.imodel3d,
                                    interptype=fit.cfg.threed.interp,
                                    smooth=fit.cfg.threed.smooth)

    lat = fit.lat3d
    lon = fit.lon3d
    
    for i in range(ncols*nrows):
        irow = i // nrows
        icol = i %  ncols
        ax = axes[irow, icol]
        # Hotspot
        if i == 0:
            # Average over all maps
            hslatavg = np.mean(
                [a.hslocbest[0] for d in fit.datasets for a in d.maps])
            hslonavg = np.mean(
                [a.hslocbest[1] for d in fit.datasets for a in d.maps])
            dist = ((fit.lat3d - hslatavg)**2 + (fit.lon3d - hslonavg)**2)**0.5
            ind  = dist.argmin()
            title = 'Hotspot'
        # Substellar point
        if i == 1:
            dist = ((fit.lat3d - 0.0)**2 + (fit.lon3d - 0.0)**2)**0.5
            ind  = dist.argmin()
            title = 'Substellar'
        # West terminator
        if i == 2:
            dist = ((fit.lat3d - 0.0)**2 + (fit.lon3d + 90.)**2)**0.5
            ind  = dist.argmin()
            title = 'West Terminator'
        # East terminator
        if i == 3:
            dist = ((fit.lat3d - 0.0)**2 + (fit.lon3d - 90.)**2)**0.5
            ind  = dist.argmin()
            title = 'East Terminator'

        tdist = tgridpost[:,:,ind]

        l1 = np.percentile(tdist, 15.87, axis=0)
        l2 = np.percentile(tdist,  2.28, axis=0)
        h1 = np.percentile(tdist, 84.13, axis=0)
        h2 = np.percentile(tdist, 97.72, axis=0)

        bf = fit.besttgrid[:,ind]

        ax.fill_betweenx(fit.p, l2, h2, facecolor='royalblue')
        ax.fill_betweenx(fit.p, l1, h1, facecolor='cornflowerblue')
        ax.semilogy(bf, fit.p, label='Best Fit', color='black')
        if irow == 1:
            ax.set_xlabel('Temperature (K)')
        if icol == 0:
            ax.set_ylabel('Pressure (bars)')
        if i == 0:
            plt.gca().invert_yaxis()

        subtitle = r'$\theta={:.2f}, \phi={:.2f}$'.format(lat[ind], lon[ind])
        ax.set_title(title + '\n' + subtitle)


    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'tgrid_unc.png'))
    plt.close(fig)

def cf_by_filter(fit, outdir=''):
    ncolumn, nlev, nfilt = fit.cf.shape
    
    ncols = np.int(np.sqrt(nfilt) // 1)
    nrows = np.int((nfilt // ncols) + (nfilt % ncols != 0))
    naxes = nrows * ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             sharey=True)
    fig.set_size_inches(8, 8)

    extra = nfilt % ncols

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
        
        for j in range(ncolumn):
                ic = fit.lon3d[j] / 360.
                if ic < 0:
                    ic += 1

                color = cmap(ic)
                label = r"${} ^\circ$".format(np.round(fit.lon3d[j], 2))
                zorder = 1
    
                ax.semilogy(fit.cf[j,:,i], fit.p, color=color,
                            label=label, zorder=zorder)

        if icol == 0:
            ax.set_ylabel('Pressure (bars)')
        if i >= naxes - ncols - (ncols - extra):
            ax.set_xlabel('Contribution (arbitrary)')

    count = 0
    for d in fit.datasets:
        for f in d.filtfiles:
            title = f.split('/')[-1]
            axes.flatten()[count].set_title(title)
            count += 1

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cf-by-filter.png'))
    plt.close(fig)

        
def cf_slice(fit, ivis=None, fname=None, outdir=''):
    """
    Slices no longer make much sense because the 3D grid is not
    rectangular. So this function needs significant revising to
    be able to function. Currently unused.
    """
    if ilat is not None and ilon is not None:
        print("Must specify either ilat or ilon, not both.")
        return

    ncolumn, nlev, nfilt = fit.cf.shape 
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

# This function doesn't work. I'll come back to it.
def clouds(fit, outdir=''):
    allrad, allmix, allq = atm.cloudmodel_to_grid(fit, fit.p,
                                                  fit.specbestp,
                                                  fit.abnbest,
                                                  fit.abnspec)

    ncloud, nlayer, nlat, nlon = allrad.shape
    
    maxrad = np.max(allrad)
    minrad = np.min(allrad[np.nonzero(allrad)])
    
    cmap = copy.copy(mpl.cm.get_cmap('hsv'))
    cmap.set_over(color='gray')

    ieq = nlat // 2

    def partrad_to_plotrad(partrad):
        s = 5 * (1 + np.log10(partrad) - np.log10(minrad))
        s = np.nan_to_num(s, neginf=0) # NaNs confuse the legend
        return s

    def plotrad_to_partrad(plotrad):
        s = 10.**(plotrad / 5.0 - 1.0 + np.log10(minrad))
        s = np.nan_to_num(s) # NaNs confuse the legend
        return s

    # This makes one call to plt.scatter() so we can easily
    # make the particle size legend.
    x = np.zeros((ncloud, nlayer, nlat, nlon))
    y = np.zeros((ncloud, nlayer, nlat, nlon))
    c = np.zeros((ncloud, nlayer, nlat, nlon))
    r = np.zeros((ncloud, nlayer, nlat, nlon))    
    for i in range(ncloud):
        for j in range(nlat):
            for k in range(nlon):
                if j == ieq:
                    ic = fit.lon[j,k] / 360.
                    if ic < 0:
                        ic += 1
                else:
                    ic = 2
                c[i,:,j,k] = ic
                x[i,:,j,k] = allmix[i,:,j,k]
                y[i,:,j,k] = fit.p
                r[i,:,j,k] = allrad[i,:,j,k]

    s = partrad_to_plotrad(r)

    # Filter out locations where there are no clouds
    x = x.flatten()
    y = y.flatten()
    s = s.flatten()
    c = c.flatten()
    wc = np.where((x != 0.0) & (s != 0.0))
    scatter = plt.scatter(x[wc], y[wc], s=s[wc], c=c[wc], cmap=cmap)

    ax = plt.gca()
    ax.set_yscale('log')
    ax.invert_yaxis()

    ax.set_ylabel('Pressure (bar)')
    ax.set_xlabel('Mixing Ratio')

    # Make room for legend, colorbar, etc
    orig_xlim = ax.get_xlim()
    orig_dx   = np.diff(orig_xlim)
    new_xlim  = (orig_xlim[0], orig_xlim[0] + orig_dx * 1.2)
    ax.set_xlim(new_xlim)

    # Longitude color bar
    cax = inset_axes(ax, width='5%', height='25%',
                     loc='lower right')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=360))
    cbar = plt.colorbar(sm, cax=cax, label=r'Longitude ($^\circ$)')
    cbar.set_ticks(np.linspace(0, 360, 5, endpoint=True))
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # Particle size legend
    nsize = len(np.unique(allrad[np.nonzero(allrad)]))
    if nsize <= 2:
        pass
    else:
        numticks = 5
        loc = mplt.LogLocator(base=10.0, numticks=numticks)
        kw = dict(prop='sizes', num=loc, color='gray',
                  func=lambda s: plotrad_to_partrad(s)) 
        pslegend = ax.legend(*scatter.legend_elements(**kw),
                             loc='upper right',
                             title="Part. Size \n($\mu$m)")

    plt.savefig(os.path.join(outdir, 'clouds.png'))

    plt.close(plt.gcf())
    

def spectra(fit, outdir=''):
    fig, ax = plt.subplots()
    fig.set_size_inches((6,8))

    cmap = copy.copy(mpl.cm.get_cmap('hsv'))

    for i in range(fit.ncolumn):
        c = fit.lon3d[i] / 360
        if c < 0:
            c += 1

        color = cmap(c)

        offset = 0.0005 * i
        
        if not np.all(fit.fluxgrid[i] == 0.0):
            ax.plot(10000/fit.modelwngrid,
                    fit.fluxgrid[i] + offset,
                    color=color)

    ax.set_ylabel(r'$F_p/F_s$ + offset')
    ax.set_xlabel(r'Wavelength ($\mu$m)')

    plt.tight_layout()

    cax = inset_axes(ax, width='30%', height='3%',
                     loc='lower right')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=360))
    cbar = plt.colorbar(sm, cax=cax, label=r'Longitude ($^\circ$)',
                        orientation='horizontal')
    cbar.set_ticks(np.linspace(0, 360, 5, endpoint=True))
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    plt.savefig(os.path.join(outdir, 'spectra.png'))
    plt.close()

def spatialsampling(fit, outdir=''):
    r = 1

    lon = np.deg2rad(fit.lon3d)
    lat = np.deg2rad(fit.lat3d)

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y , z, color='black', s=20, zorder=2, label='Cell centers')

    # Wireframe
    latwf, lonwf = np.meshgrid(np.linspace( -90.,  90., 20, endpoint=True),
                               np.linspace(-180., 180., 40, endpoint=True),
                               indexing='ij')

    lonwf = np.deg2rad(lonwf)
    latwf = np.deg2rad(latwf)

    xwf = r * np.cos(latwf) * np.cos(lonwf)
    ywf = r * np.cos(latwf) * np.sin(lonwf)
    zwf = r * np.sin(latwf)

    ax.plot_wireframe(xwf, ywf, zwf, alpha=0.2, zorder=1)

    ax.set_xlabel(r'x ($R_p$)')
    ax.set_ylabel(r'y ($R_p$)')
    ax.set_zlabel(r'z ($R_p$)')

    ax.legend()

    plt.savefig(os.path.join(outdir, 'spatialsampling.png'))
    plt.close()

def abundances(fit, outdir=''):
    fig, ax = plt.subplots(figsize=(6,10))

    dist = (fit.lat3d**2 + fit.lon3d**2)**0.5
    iss = np.where(dist == np.min(np.abs(dist)))[0][0]

    for mol in fit.cfg.threed.mols:
        ind = np.where(fit.abnspec == mol)[0][0]
        plt.loglog(fit.abnbest[ind,:,iss], fit.p, label=mol)

    ax.set_title('Substellar point abundances')

    ax.invert_yaxis()

    ax.set_xlim((10**-12, 1))
        
    ax.legend()    
    plt.savefig(os.path.join(outdir, 'abundances.png'))
    plt.close()
    
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
