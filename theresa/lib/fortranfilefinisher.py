'''
File taken from Emily Rauscher's github, 2020-12-9. Credit for
original development goes to Erin May. Modifications made
by Ryan Challener.
'''

import numpy as np
import math
import pickle

from scipy.io import readsav

def get_sigma(oom,nlev):
    sigma=np.empty([nlev])*0.0
    if oom==0:
        stp=1.0/(nlev+1.)
        sigma[nlev-1]=1.0-stp
        for n in range(nlev-2,-1,-1):
            sigma[n]=sigma[n+1]-stp
    if oom>0:
        stp=-1.0*oom/nlev
        sigma[nlev-1]=10.**(stp/2.)
        for n in range(nlev-2,-1,-1):
            sigma[n]=sigma[n+1]*10.**(stp)       
    return sigma

####################

def fort26(path,runname,oom, surfp,ver,savet,fortfile,):
    
    with open(path+runname+'/'+fortfile) as f:
        first_line=f.readline()
        nlat,nlon,nlev=first_line.split()
        nlat,nlon,nlev=int(nlat),int(nlon),int(nlev)
        print('  ')
        print(' ....reading ',fortfile)
        print('       nlat=', nlat, 'nlon=', nlon, 'nlev=', nlev)
    f.close()
    
    data26=np.empty([nlon*nlat*nlev, 6])   
    
    l=0
    lp=0
    with open(path+runname+'/'+fortfile) as f:
        for line in f:
            if l==0:
                l+=1
                continue
            elif l%2==1 and l<=nlon*nlat*nlev*2.:
                line_pair=np.empty([6])
                lon, lat, lev, u, v = line.split()
                line_pair[:5] = np.float32(lon), np.float32(lat), int(lev), np.float32(u), np.float32(v)
            elif l%2==0 and l<=nlon*nlat*nlev*2.:
                line_pair[5]=np.float32(line)
                data26[lp,:]=line_pair
                lp+=1
            elif l>nlon*nlat*nlev*2.:
                print('       END OF FILE: DONE')
                break
            l+=1
    f.close()

    lon_arr_f=data26[:,0]
    lon_arr=np.array([])
    for l in range(0,len(lon_arr_f)):
        el=lon_arr_f[l]
        if not el in lon_arr:
            lon_arr=np.append(lon_arr,el)

    lat_arr_f=data26[:,1]
    lat_arr=np.array([])
    for l in range(0,len(lat_arr_f)):
        el=lat_arr_f[l]
        if not el in lat_arr:
            lat_arr=np.append(lat_arr,el)

    lev_arr_f=data26[:,2]
    lev_arr=np.array([])
    for l in range(0,len(lev_arr_f)):
        el=lev_arr_f[l]
        if not el in lev_arr:
            lev_arr=np.append(lev_arr,el)

    data_26=np.empty([nlev,nlon,nlat,6])
    for l in range(0,data26.shape[0]):
        lon,lat,lev=data26[l,:3]
        lon_i,lat_i,lev_i=np.where(lon_arr==lon)[0][0],np.where(lat_arr==lat)[0][0],np.where(lev_arr==lev)[0][0]
        data_26[lev_i,lon_i,lat_i,:]=data26[l,:]
    ############################################################
    nlev,nlon,nlat,nparam=data_26.shape
    
    if ver==True:
        print(' ')
        print('--------------------------')
        print('|    ARRAY DIMENSIONS    |')
        print('--------------------------')
        print('N_levels: ', nlev)
        print('N_lons:   ', nlon)
        print('N_lats:   ', nlat)
        print('N_params: ', nparam)
    
        print(' ')
        print('--------------------------')
        print('|   Important Indexing!  |')
        print('--------------------------')
        print('Longitudes: 0')
        print('Latitudes : 1')
        print('Levels    : 2')
        print('U Winds   : 3')
        print('V Winds   : 4')
        print('Temps     : 5')
    # nparam index: 
    #      0=lons
    #      1=lats
    #      2=levs
    #      3=u wind
    #      4=v wind
    #      5=temps
            
    p_BAR=get_sigma(oom,nlev)*surfp
    
    if ver==True:
        print(' ')
        print('PRESSURE ARRAY: ')
        print(p_BAR)  

    lat_arr=data_26[0,0,:,1]
    lon_arr=data_26[0,:,0,0]

    if ver==True:
        print(' ')
        print('LATITUDE ARRAY: ')
        print(lat_arr)
        print(' ')
        print('LONGITUDE ARRAY: ')
        print(lon_arr)  
        
    if savet==True:
        pickle.dump([p_BAR,lon_arr,lat_arr],open(path+runname+'/pres_lon_lat.txt','wb'))
        pickle.dump(data_26,open(path+runname+'/fort26.txt','wb'))
    
    return runname,oom,surfp,lon_arr,lat_arr,p_BAR,data_26

####################################

def fort52(path,runname,oom,surfp,nlev,nday,savet,fortfile):
    
    with open(path+runname+'/'+fortfile,'r') as data_52:
        ke=[] #kinetic energy 
        cpt=np.zeros((nlev,nday+91))*np.nan #kinetic energy and cpt
        dayval=[] #column corresponding to the day 
        daycount=0
        laycount=0
        for line in data_52:
            p=line.split()
            #print laycount,daycount
            if laycount < nlev-1: 
                ke.append(float(p[0]))
                if daycount != nday+91:
                    cpt[laycount,daycount]=(float(p[0]))
                    dayval.append(daycount)
                    laycount=laycount+1
            else:
                laycount=0
                ke.append(float(p[0]))
                if daycount != nday+91:
                    cpt[laycount,daycount]=(float(p[0]))
                    dayval.append(daycount)
                    daycount=daycount+1

    cpt=cpt*surfp*100000.

    p_BAR=get_sigma(oom,nlev)*surfp
    cpt=cpt+.01
    cpt=np.log(cpt)
    daylist=np.arange(3,nday+1,1)
    #print len(daylist)
    
    newke=cpt[:,3:nday+1]
    
    if savet==True:
        pickle.dump(newke,open(path+runname+'/fort52.txt','wb'))
    
    return runname,oom,surfp,daylist,p_BAR,newke

####################################

def fort64(path,runname,radea,ver,savet,fortfile):
    
    with open(path+runname+'/'+fortfile) as f:
        first_line=f.readline()
        nlat,nlon,nlev=first_line.split()
        nlat,nlon,nlev=int(nlat),int(nlon),int(nlev)
        print('  ')
        print(' ....reading fort.64 (LW)')
        print('       nlat=', nlat, 'nlon=', nlon, 'nlev=', nlev)

        data_64=np.empty([nlat*nlon,3])*0.0
        l=0
        with open(path+runname+'/'+fortfile) as f:
            for line in f:
                if l==0:
                    l+=1
                    continue
                if l>nlat*nlon:
                    continue
                    l+=1
                data_64[l-1] = line.split()
                l+=1
            print('       END OF FILE: DONE')
        f.close()

        lon_arr_f=data_64[:,0]
        lon_arr=np.array([])
        l=0
        while l<data_64.shape[0]:
            lon_arr=np.append(lon_arr,lon_arr_f[l])
            l+=nlat

        lat_arr=data_64[:nlat,1]

        data_lw=np.empty([nlon,nlat,3])
        for l in range(0,data_64.shape[0]):
            lon,lat=data_64[l,:2]
            lon_i,lat_i=np.where(lon_arr==lon)[0][0],np.where(lat_arr==lat)[0][0]
            data_lw[lon_i,lat_i,:]=data_64[l,:]
            
        data_lw=data_lw[:,:,2]
        
        
        
    total_lw=np.nansum(data_lw[:,:]*np.cos(lat_arr*np.pi/180.))*(2*np.pi/nlon)*(np.pi/nlat)*radea**2.
    dayside_lw=((np.nansum(data_lw[0:np.int(nlon/4),:]*np.cos(lat_arr*np.pi/180.))  
                                      +np.nansum(data_lw[np.int(nlon*3./4):nlon-1,:]*np.cos(lat_arr*np.pi/180.))) 
                                    *(2.*np.pi/nlon)*(np.pi/nlat)*radea**2)
    if ver==True:
        print('******************************')
        print('Total Integrated Output (W):')
        print('  LW:', total_lw)
        print('-------------------------------')
        print(' Dayside Integrated Output (W):')
        print('  LW:', dayside_lw)
        print('******************************')

    if savet==True:
        pickle.dump(data_lw,open(path+runname+'/fort64.txt','wb'))
        pickle.dump(total_lw,open(path+runname+'/LWF_total.txt', 'wb'))
    
    return runname,lon_arr,lat_arr,data_lw,total_lw

####################################

def fort65(path,runname,radea,ver,savet,fortfile):
    
    with open(path+runname+'/'+fortfile) as f:
        first_line=f.readline()
        nlat,nlon,nlev=first_line.split()
        nlat,nlon,nlev=int(nlat),int(nlon),int(nlev)
        print('  ')
        print(' ....reading fort.65 (SW)')
        print('       nlat=', nlat, 'nlon=', nlon, 'nlev=', nlev)


    data_65=np.empty([nlat*nlon,3])*0.0
    l=0
    with open(path+runname+'/'+fortfile) as f:
        for line in f:
            if l==0:
                l+=1
                continue
            if l>nlat*nlon:
                continue
                l+=1
            #print line, line.split()
            data_65[l-1] = line.split()
            l+=1
        print('       END OF FILE: DONE')
    f.close()


    lon_arr_f=data_65[:,0]
    lon_arr=np.array([])
    l=0
    while l<data_65.shape[0]:
        lon_arr=np.append(lon_arr,lon_arr_f[l])
        l+=nlat

    lat_arr=data_65[:nlat,1]

    data_sw=np.empty([nlon,nlat,3])
    for l in range(0,data_65.shape[0]):
        lon,lat=data_65[l,:2]
        lon_i,lat_i=np.where(lon_arr==lon)[0][0],np.where(lat_arr==lat)[0][0]
        data_sw[lon_i,lat_i,:]=data_65[l,:]

    data_sw=data_sw[:,:,2]
    
    total_sw=np.nansum(data_sw[:,:]*np.cos(lat_arr*np.pi/180.))*(2*np.pi/nlon)*(np.pi/nlat)*radea**2.
    dayside_sw=((np.nansum(data_sw[0:np.int(nlon/4),:]*np.cos(lat_arr*np.pi/180.))  
                                      +np.nansum(data_sw[np.int(nlon*3./4):nlon-1,:]*np.cos(lat_arr*np.pi/180.))) 
                                    *(2.*np.pi/nlon)*(np.pi/nlat)*radea**2)
    if ver==True:
        print('******************************')
        print('Total Integrated Output (W):')
        print('  SW:', total_sw)
        print('-------------------------------')
        print(' Dayside Integrated Output (W):')
        print('  SW:', dayside_sw)
        print('******************************')

    if savet==True:
        pickle.dump(data_sw,open(path+runname+'/fort65.txt','wb'))
        pickle.dump(total_sw,open(path+runname+'/SWF_total.txt', 'wb'))
   
    
    return runname,lon_arr,lat_arr,data_sw,total_sw

####################################

def fort66(path,runname,radea,ver,savet,fortfile):
    with open(path+runname+'/'+fortfile) as f:
        first_line=f.readline()
        nlat,nlon,nlev=first_line.split()
        nlat,nlon,nlev=int(nlat),int(nlon),int(nlev)
        print('  ')
        print(' ....reading fort.66 (Total)')
        print('       nlat=', nlat, 'nlon=', nlon, 'nlev=', nlev)


    data_66=np.empty([nlat*nlon,3])*0.0
    l=0
    with open(path+runname+'/'+fortfile) as f:
        for line in f:
            if l==0:
                l+=1
                continue
            if l>nlat*nlon:
                continue
                l+=1
            #print line, line.split()
            data_66[l-1] = line.split()
            l+=1
        print('       END OF FILE: DONE')
    f.close()

    lon_arr_f=data_66[:,0]
    lon_arr=np.array([])
    l=0
    while l<data_66.shape[0]:
        lon_arr=np.append(lon_arr,lon_arr_f[l])
        l+=nlat

    lat_arr=data_66[:nlat,1]

    data_tot=np.empty([nlon,nlat,3])
    for l in range(0,data_66.shape[0]):
        lon,lat=data_66[l,:2]
        lon_i,lat_i=np.where(lon_arr==lon)[0][0],np.where(lat_arr==lat)[0][0]
        data_tot[lon_i,lat_i,:]=data_66[l,:]

    data_tot=data_tot[:,:,2]
    
    total_tt=np.nansum(data_tot[:,:]*np.cos(lat_arr*np.pi/180.))*(2*np.pi/nlon)*(np.pi/nlat)*radea**2.
    dayside_tt=((np.nansum(data_tot[0:np.int(nlon/4),:]*np.cos(lat_arr*np.pi/180.))  
                                      +np.nansum(data_tot[np.int(nlon*3./4):nlon-1,:]*np.cos(lat_arr*np.pi/180.))) 
                                    *(2.*np.pi/nlon)*(np.pi/nlat)*radea**2)
    if ver==True:
        print('******************************')
        print('Total Integrated Output (W):')
        print('  TT:', total_tt)
        print('-------------------------------')
        print(' Dayside Integrated Output (W):')
        print('  TT:', dayside_tt)
        print('******************************')

    if savet==True:
        pickle.dump(data_tot,open(path+runname+'/fort66.txt','wb'))
        pickle.dump(total_tt,open(path+runname+'/TTF_total.txt', 'wb'))
    
    return runname,lon_arr,lat_arr,data_tot,total_tt
