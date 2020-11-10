import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        ax.set_title('{} um'.format(wl[i]))

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-{}-maps.png'.format(proj)))
    plt.close(fig)

def bestfit(t, model, data, unc, wl, outdir):
    fig, ax = plt.subplots()

    nt = len(t)
    
    for i in range(len(wl)):
        ax.plot(t, model[i*nt:(i+1)*nt], zorder=2)
        ax.errorbar(t, data[i], unc[i], zorder=1)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized Flux')
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-lcs.png'))
    plt.close(fig)

def bestfitlcsspec(fit):
    fig, ax = plt.subplots()

    nfilt, nt = fit.specbestmodel.shape

    for i in range(nfilt):
        ax.scatter(fit.t, fit.flux[i], s=0.1, zorder=1)
        ax.plot(fit.t, fit.specbestmodel[i], zorder=2)

    ax.set_ylabel('Fs/Fp')
    ax.set_xlabel('Wavelength (um)')
    plt.tight_layout()
    plt.savefig(os.path.join(fit.cfg.outdir, 'bestfit-lcs-spec.png'))
    plt.close(fig)

def bestfittgrid(fit):
    fig, ax = plt.subplots()

    for i in range(fit.cfg.res):
        for j in range(fit.cfg.res):
            lat = fit.lat[i,j]
            lon = fit.lon[i,j]
            label = "Lat: {:.1f}, Lon: {:.1f}".format(lat, lon)
            if ((lon + fit.dlon < fit.minvislon) or
                (lon - fit.dlon > fit.maxvislon)):
                linestyle = '--'
            else:
                linestyle = '-'
                ax.semilogy(fit.besttgrid[:,i,j], fit.p, label=label,
                            linestyle=linestyle)

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
        print(j)
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
            ax.set_title('{:.1f} um'.format(fit.wlmid[i]))
            frame_ims.append(im)
            
        all_ims.append(frame_ims)
   
    ani = animation.ArtistAnimation(fig, all_ims, interval=50,
                                    blit=True, repeat_delay=1000)

    ani.save(os.path.join(fit.cfg.outdir, 'fmaps.gif'), dpi=300, writer=writer)

    plt.close(fig)
