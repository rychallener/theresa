import os
import numpy as np
import matplotlib.pyplot as plt

def circmaps(planet, eigeny, outdir):
    nharm, ny = eigeny.shape

    for j in range(nharm):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1

        fill = np.int(np.log10(nharm)) + 1
        fnum = str(j).zfill(fill)
        
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.imshow(planet.map.render(theta=180).eval(),
                  origin="lower", cmap="plasma")
        ax.axis("off")
        plt.savefig(os.path.join(outdir, 'emap-ecl-{}.png'.format(fnum)))
        plt.close(fig)

def rectmaps(planet, eigeny, outdir):
    nharm, ny = eigeny.shape

    for j in range(nharm):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j, yi]
                yi += 1

        fill = np.int(np.log10(nharm)) + 1
        fnum = str(j).zfill(fill)    
        fig, ax = plt.subplots(1, figsize=(6,3))
        ax.imshow(planet.map.render(projection="rect").eval(),
                  origin="lower", cmap="plasma")
        plt.savefig(os.path.join(outdir, 'emap-rect-{}.png'.format(fnum)))
        plt.close(fig)

def lightcurves(t, lcs, outdir):
    nharm, nt = lcs.shape
    
    l =  1
    m = -1
    pos = True

    fig, ax = plt.subplots(1, figsize=(6,3))
    
    for i in range(nharm):
        print(l, m, pos)
        plt.plot(t, lcs[i], label=r"${}Y_{{{}{}}}$".format(["-", "+"][pos],
                                                           l, m))
        if pos:
            pos = False
        else:
            if l == m:
                l += 1
                m  = -l
                pos = True
            else:
                pos = True
                m += 1
            
    plt.ylabel('Time (days)')
    plt.xlabel('Normalized Flux')
    plt.legend(ncol=l, fontsize=6)
    plt.savefig(os.path.join(outdir, 'lightcurves.png'))
    
