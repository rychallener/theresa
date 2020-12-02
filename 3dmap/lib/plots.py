import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rcParams['axes.formatter.useoffset'] = False

def circmaps(planet, eigeny, outdir, ncurves=None):    
    nharm, ny = eigeny.shape

    if type(ncurves) == type(None):
        ncurves = nharm

    lmax = np.int((nharm / 2 + 1)**0.5 - 1)

    for j in range(ncurves):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1

        fill = np.int(np.log10(ncurves)) + 1
        fnum = str(j).zfill(fill)
        
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.imshow(planet.map.render(theta=180).eval(),
                  origin="lower", cmap="plasma")
        ax.axis("off")
        plt.savefig(os.path.join(outdir, 'emap-ecl-{}.png'.format(fnum)))
        plt.close(fig)

def rectmaps(planet, eigeny, outdir, ncurves=None):
    nharm, ny = eigeny.shape

    if type(ncurves) == type(None):
        ncurves = nharm
    
    lmax = np.int((nharm / 2 + 1)**0.5 - 1)
    
    for j in range(ncurves):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1

        fill = np.int(np.log10(ncurves)) + 1
        fnum = str(j).zfill(fill)    
        fig, ax = plt.subplots(1, figsize=(6,3))
        ax.imshow(planet.map.render(theta=180, projection="rect").eval(),
                  origin="lower", cmap="plasma",
                  extent=(-180, 180, -90, 90))
        plt.savefig(os.path.join(outdir, 'emap-rect-{}.png'.format(fnum)))
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
    
def ecurvepower(evalues, outdir):
    ncurves = len(evalues)
    num = np.arange(1, ncurves + 1)

    fig, ax = plt.subplots()
    
    plt.plot(num, evalues / np.sum(evalues), 'ob')
    plt.xlabel('E-curve Number')
    plt.ylabel('Normalized Power')

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'ecurvepower.png'))

def pltmaps(maps, wl, outdir, proj='rect'):
    nmaps = len(wl)

    ncols = np.min((nmaps, 3))
    nrows = nmaps // ncols + (nmaps % ncols != 0)

    xsize = 7. / 3. * ncols
    if proj == 'rect':
        ysize = 7. / 3. / 2. * nrows
    elif proj == 'ortho':
        ysize = 7. / 3. * nrows

    figsize = (xsize, ysize)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             sharey=True, squeeze=False, figsize=figsize)

    vmax = np.max(maps[~np.isnan(maps)])
    vmin = np.min(maps[~np.isnan(maps)])
    
    if proj == 'rect':
        extent = (-180, 180, -90, 90)
    elif proj == 'ortho':
        extent = (-90,   90, -90, 90)

    for i in range(nmaps):
        irow = i // ncols
        icol = i %  ncols
        ax = axes[irow,icol]
        im = ax.imshow(maps[i], origin='lower', cmap='plasma', extent=extent,
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title('{:.2f} um'.format(wl[i]))

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-{}-maps.png'.format(proj)))
    plt.close(fig)

def bestfit(t, model, data, unc, wl, outdir):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nfilt = len(wl)
    nt = len(t)

    hratios = np.zeros(nfilt+1)
    hratios[0] = 0.5
    hratios[1:] = 0.5 / nfilt
    
    gridspec_kw = {'height_ratios':hratios}
    
    fig, axes = plt.subplots(nrows=nfilt+1, ncols=1, sharex=True,
                             gridspec_kw=gridspec_kw, figsize=(8,10))

    nt = len(t)
    
    for i in range(nfilt):
        axes[0].plot(t, model[i*nt:(i+1)*nt], zorder=2, color=colors[i],
                     label='{:.2f} um'.format(wl[i]))
        axes[0].scatter(t, data[i], s=0.1, zorder=1, color=colors[i])

    axes[0].legend()
    axes[0].set_ylabel(r'($F_s + F_p$)/$F_s$')

    for i in range(nfilt):
        axes[i+1].scatter(t, data[i] - model[i*nt:(i+1)*nt], s=0.1,
                          color=colors[i])
        axes[i+1].set_ylabel('Residuals')
        axes[i+1].axhline(0, 0, 1, color='black', linestyle='--')
        if i == nfilt-1:
            axes[i+1].set_xlabel('Time (days)')

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-lcs.png'))
    plt.close(fig)

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
    fig, ax = plt.subplots()

    # Latitude index 
    ieq = fit.cfg.res // 2

    for i in range(fit.cfg.res):
        for j in range(fit.cfg.res):
            lat = fit.lat[i,j]
            lon = fit.lon[i,j]
            if i == ieq:
                label = "Lat: {:.1f}, Lon: {:.1f}".format(lat, lon)
                color = None
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
        ilat = cfg.res // 2
    if type(ilon) == type(None):
        ilon = cfg.res // 2
        
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
    transform = matplotlib.transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    # Note: assumes all filters are normalized to 1, and plots them
    # in the top tenth of the image.
    for i in range(nfilt):
        plt.plot(np.log10(fit.filtwl[i]), 1.0 - fit.filttrans[i]/10.,
                 transform=transform, label='{:.2f} um'.format(fit.wlmid[i]),
                 linestyle='--')

    plt.legend(frameon=False)
    plt.colorbar()
    plt.savefig(os.path.join(fit.cfg.outdir, 'cf.png'))
    plt.close()
