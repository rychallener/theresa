import os
import numpy as np
import matplotlib.pyplot as plt

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

def mapsumcirc(planet, eigeny, params, outdir, ncurves=None, res=300):
    if type(ncurves) == type(None):
        ncurves = eigeny.shape[0]
        
    # Reset planet's harmonic coefficients
    planet.map[1:,:] = 0

    # Start with uniform map with correct total flux
    map = planet.map.render(theta=180, res=res).eval() * params[ncurves]
    
    fig, ax = plt.subplots()
    
    for i in range(ncurves):
        planet.map[1:,:] = eigeny[i,1:]
        map += params[i] * planet.map.render(theta=180, res=res).eval()

    im = ax.imshow(map, origin='lower', cmap='plasma')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-ecl.png'))        
    plt.close(fig)

def mapsumrect(planet, eigeny, params, outdir, ncurves=None, res=300):
    if type(ncurves) == type(None):
        ncurves = eigeny.shape[0]

    # Reset planet's harmonic coefficients
    planet.map[1:,:] = 0

    # Start with uniform map with correct total flux
    map = planet.map.render(theta=180, res=res,
                               projection='rect').eval() * params[ncurves]
    
    fig, ax = plt.subplots()
    
    for i in range(ncurves):
        planet.map[1:,:] = eigeny[i,1:]
        map += params[i] * planet.map.render(theta=180, res=res,
                                             projection='rect').eval()

    im = ax.imshow(map, origin='lower', cmap='plasma',
                   extent=(-180, 180, -90, 90))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-rect.png'))        
    plt.close(fig)

def bestfit(t, model, data, unc, outdir):
    fig, ax = plt.subplots()
    ax.plot(t, model, zorder=2)
    ax.errorbar(t, data, unc, zorder=1)
    ax.set_xlabel('Time (days)')

    ax.set_ylabel('Normalized Flux')
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'bestfit-lc.png'))
    plt.close(fig)
    

    