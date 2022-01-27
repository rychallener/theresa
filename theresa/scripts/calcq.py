#! /usr/bin/env python3

import numpy as np

def readcloudsfile(fname):
    '''
    Format is this:
    4 rows of comments
    A row of nlat, nlon, nlayers
    A row of lat, lon, level, alt (m), p (bar), T (K), t_std (K),
    EW vel (m/s), NS vel (m/s), vert vel (m/s), clouds 1-13 vis tau (650 nm),
    cloud total vis tau, cloud 1-13 IR tau (5 um), cloud total IR tau
    This last row spills over onto the next line for the last 3 entries. Tragic.
    '''
    ncomment = 4
    ncol = 38
    nextra = 3

    # Get species names
    with open(fname, 'r') as f:
        next(f)
        namestring = f.readline()
        spec = namestring.split(':')[1].split(',')
        spec = [x.strip((' \n')) for x in spec]
    
    with open(fname, 'r') as f:
        for i in range(ncomment):
            next(f)
        nlat, nlon, nlayer = [int(a) for a in f.readline().split()]

        data = np.zeros((nlat * nlon * nlayer, ncol))

        for i, line in enumerate(f.readlines()):
            if i % 2 == 0:
                data[i//2,:ncol-nextra] = [float(a) for a in line.split()]
            else:
                data[i//2,ncol-nextra:] = [float(a) for a in line.split()] 

    lat_all = data[:,0]
    lon_all = data[:,1]
    lev_all = data[:,2]

    lat = np.unique(data[:,0])
    lon = np.unique(data[:,1])
    lev = np.unique(data[:,2])

    data2 = np.zeros((nlat, nlon, nlayer, ncol - 3))

    for i in range(data.shape[0]):
        tlat, tlon, tlev = data[i,:3]
        ilat = np.where(tlat == lat)
        ilon = np.where(tlon == lon)
        ilev = np.where(tlev == lev)
        data2[ilat, ilon, ilev] = data[i,3:]

    return lat, lon, lev, spec, data2

def calcq(taufname, massfname, partsizefname, densfname, r0):
    '''
    Calculates Q used by Lee et al. 2013 from a cloud report file.
    WARNING: the cloud report files have a very specific format
    that is assumed by this function. Be very careful.
    '''
    # Read files
    # massdata is the same as taudata but the vis tau columns have been
    # replaced with cloud mass in kg
    lat, lon, lev, spec, taudata  = readcloudsfile(taufname)
    lat, lon, lev, spec, massdata = readcloudsfile(massfname)

    nlat = len(lat)
    nlon = len(lon)
    nlev = len(lev)

    # Particle size (from Michael)
    a = np.loadtxt(partsizefname)
    # Densities (Roman et al 2021)
    rho = np.loadtxt(densfname)

    # Calculate Qext
    # tau = Q * pi * a**2 * n * delz
    # Q = tau / (pi * a**2 * n * delz)
    Rjup = 6.9911e7
    ncloud = 13
    V    = np.zeros((nlat, nlon, nlev))
    A    = np.zeros((nlat, nlon, nlev))
    n    = np.zeros((nlat, nlon, nlev, ncloud))
    Qext = np.zeros((nlat, nlon, nlev, ncloud))
    Q    = np.zeros((nlat, nlon, nlev, ncloud))
    for ilat in range(nlat):
        for ilon in range(nlon):
            for ilev in range(nlev):
                r  = taudata[ilat,ilon,ilev,0] + r0
                # These are not-so-great assumptions of cell sizes. Should ask
                # Michael if he has exact numbers.
                if ilev == 0:
                    ri = r - (taudata[ilat,ilon,ilev,  0] - \
                              taudata[ilat,ilon,ilev+1,0]) / 2.
                    rf = r + (taudata[ilat,ilon,ilev,  0] - \
                              taudata[ilat,ilon,ilev+1,0]) / 2.
                elif ilev == nlev - 1:
                    ri = r
                    rf = r + (taudata[ilat,ilon,ilev-1,0] - \
                              taudata[ilat,ilon,ilev,  0]) / 2.
                else:
                    ri = r - (taudata[ilat,ilon,ilev,  0] - \
                              taudata[ilat,ilon,ilev+1,0]) / 2.
                    rf = r + (taudata[ilat,ilon,ilev-1,0] - \
                              taudata[ilat,ilon,ilev,  0]) / 2.

                if ilon == 0:
                    phii = lon[ilon] - (lon[ilon+1] - lon[ilon  ]) / 2.
                    phif = lon[ilon] + (lon[ilon+1] - lon[ilon  ]) / 2.
                elif ilon == nlon - 1:
                    phii = lon[ilon] - (lon[ilon  ] - lon[ilon-1]) / 2.
                    phif = lon[ilon] + (lon[ilon  ] - lon[ilon-1]) / 2.
                else:
                    phii = lon[ilon] - (lon[ilon  ] - lon[ilon-1]) / 2.
                    phif = lon[ilon] + (lon[ilon+1] - lon[ilon  ]) / 2.

                if ilat == 0:
                    thetai = -90.
                    thetaf = lat[ilat] + (lat[ilat+1] - lat[ilat  ]) / 2.
                elif ilat == nlat - 1:
                    thetai = lat[ilat] - (lat[ilat  ] - lat[ilat-1]) / 2.
                    thetaf = 90.
                else:
                    thetai = lat[ilat] - (lat[ilat  ] - lat[ilat-1]) / 2.
                    thetaf = lat[ilat] + (lat[ilat+1] - lat[ilat  ]) / 2.

                # Volume of the cell
                V[ilat,ilon,ilev] = (rf**3 - ri**3) / 3 * \
                    (np.sin(np.deg2rad(thetaf)) - np.sin(np.deg2rad(thetai))) * \
                    (np.deg2rad(phif) - np.deg2rad(phii))
                # Area of the (base of the) cell
                A[ilat,ilon,ilev] = ri**2 * \
                    (np.sin(np.deg2rad(thetaf)) - np.sin(np.deg2rad(thetai))) * \
                    (np.deg2rad(phif) - np.deg2rad(phii))                

                for icloud in range(ncloud):
                    # I think mass is per unit area?
                    #n  = 3. / 4. * massdata[ilat,ilon,ilev,7+icloud] / \
                    #    (rf - ri) / (rho[icloud] * np.pi * a[ilev]**3)
                    n[ilat,ilon,ilev,icloud] = \
                        3. / 4. * massdata[ilat,ilon,ilev,7+icloud] * \
                        A[ilat,ilon,ilev] / \
                        (rho[icloud] * np.pi * a[ilev]**3. * V[ilat,ilon,ilev])

                    #n[ilat,ilon,ilev,icloud] = \
                    #    3. / 4. * massdata[ilat,ilon,ilev,7+icloud] / \
                    #    (rho[icloud] * np.pi * a[ilev]**3)

                    if n[ilat,ilon,ilev,icloud] != 0.0:
                        Qext[ilat,ilon,ilev,icloud] = \
                            taudata[ilat,ilon,ilev,21+icloud] / \
                            (np.pi * a[ilev]**2 * \
                             n[ilat,ilon,ilev,icloud] * (rf - ri))

                        # Convert to Q, assuming 5 um
                        x = 2 * np.pi * a[ilev] / 5e-6
                        Q[ilat,ilon,ilev,icloud] = \
                            (5 / Qext[ilat,ilon,ilev,icloud] - x**0.2) * x**4
                    # This ensures that the optical depth will be
                    # zero, or very close to zero, in grid cells where
                    # the GCM believes there is no
                    # condensation. Otherwise, if these Qs are used in
                    # conjunction with a different chemistry
                    # prescription (e.g., equilibrium condensation),
                    # there may be optical depth where none is
                    # expected and such a forward model would give
                    # much different results than the GCM.
                    else:
                        Q[ilat,ilon,ilev,icloud] = 1e300

    return Q, n, spec
