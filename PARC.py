#!/usr/bin/env python

# Import packages
import numpy as np
import vpython as vp
import urllib
from multiprocessing import Pool, cpu_count

# Set Number of CPUs the run on, default is maximum available
cpus=cpu_count()

# Constants and Unit Conversion

G=6.67408*10**(-11) # Newton's Gravity Constant (m^3 kg^-1 s^-2)
Msun=1.988*10**30 # Mass of the Sun (kg)
u=Msun*G
MEarth=5.9722*10**24 # Mass of Earth (kg)
REarth=6378000 # Orbital Radius of earth (meters)
ue=MEarth*G

# Define Mars Parameters

Marsi=1.85061 # Mars Orbital Inclination (deg)
MarsO=49.57854 # Mars Longitude of Ascending Node (deg)
Marslongperi=336.04084 # Mars Longitude of Periapsis (deg)
Marsw=Marslongperi-MarsO # Mars Argument of Periapsis (deg)
Marsa=1.52366231 # Mars Semi-major axis (AU)
Marsec=0.09341233 # Mars Orbital Eccentricty
MMars=6.4169E23 # Mars Mass (kg)
um=MMars*G
Phobosa=9373000 # Phobos semi-major axis in orbit around Mars (meters)
VMarspark=np.sqrt(um*(2/Phobosa-1/Phobosa)) # Mars centric orbital velocity at parking orbit
VMarsesc=np.sqrt(2*um/Phobosa) # escape velocity at parking orbit

#Magnitude to Diameter Scalings from MPC https://www.minorplanetcenter.net/iau/Sizes.html

magnitudes=np.asarray(range(100))/2-2
# for albedo 0.25:
diameters_base025=np.asarray([6700,5300,4200,3300,2600,2100,1700,1300,1050,840,670,530,420,330,260,210,170,130,110,85,65,50,40,35,25,20,17,13,11,8])
diameters025=np.concatenate((diameters_base025*1e3,diameters_base025,diameters_base025/1e3,np.asarray([7,5,4,3,3,2,2,1,1,1])/1e3))
# for albedo 0.05
diameters_base005=np.asarray([14900,11800,9400,7500,5900,4700,3700,3000,2400,1900,1500,1200,940,740,590,470,370,300,240,190,150,120,95,75,60,50,37,30,24,19])
diameters005=np.concatenate((diameters_base005*1e3,diameters_base005,diameters_base005/1e3,np.asarray([15,12,9,7,6,5,4,3,2,2])/1e3))
mag2diameter=np.hstack((magnitudes[:,np.newaxis], diameters025[:,np.newaxis], diameters005[:,np.newaxis]))


#Keplerian and Cartesian Conversion Functions

def kep2cart(O,w,i,ec,a,nu): 
    # converts orbital elements to cartesian position and velocity vectors, give inputs in deg or AU
    w=vp.radians(w)
    i=vp.radians(i)
    O=vp.radians(O)
    nu=vp.radians(nu)
    a=a/6.68459E-12
    r=a*(1-ec**2)/(1+ec*np.cos(nu))
    h=np.sqrt(u*a*(1-ec**2))
    p=a*(1-ec**2)
    pos=r*vp.vector(np.cos(O)*np.cos(w+nu)-np.sin(O)*np.sin(w+nu)*np.cos(i),
                    np.sin(O)*np.cos(w+nu)+np.cos(O)*np.sin(w+nu)*np.cos(i),
                    np.sin(i)*np.sin(w+nu))
    velo=vp.vector(h/r*(pos.x*ec/p*np.sin(nu)-(np.cos(O)*np.sin(w+nu)+np.sin(O)*np.cos(w+nu)*np.cos(i))),
                   h/r*(pos.y*ec/p*np.sin(nu)-(np.sin(O)*np.sin(w+nu)-np.cos(O)*np.cos(w+nu)*np.cos(i))),
                   h/r*(pos.z*ec/p*np.sin(nu)+np.cos(w+nu)*np.sin(i)))
    return pos, velo

def cart2kep(r,v):
    # converts cartesian coordinates to orbital elements, all inputs in m and m/s, outputs are in radians and AU, r is the periapsis position vector, v is the velocity vector at periapsis
    h=vp.cross(r,v) # orbital momentum vector, perpendicular to orbital plane
    evec=(vp.cross(v,h)/u)-vp.norm(r) #eccentricity vector
    n=vp.cross(vp.vector(0,0,1),h) #vector to ascending node
    ec=vp.mag(evec)
    i=vp.diff_angle(h,vp.vector(0,0,1)) 
    if n.y>=0: O=vp.diff_angle(n,vp.vector(1,0,0))
    elif n.y<0: O=2*np.pi-vp.diff_angle(n,vp.vector(1,0,0))
    if r.z>=0: w=vp.diff_angle(n,r)
    elif r.z<0: w=2*np.pi-vp.diff_angle(n,r)
    a=1/((2/vp.mag(r))-(vp.mag2(v)/u))
    if i==0: # Sets poorly defined longitude of ascending node as argument of periapsis for an orbit with 0 inclination
        O=0
        if r.y>=0: w=vp.diff_angle(vp.vector(1,0,0),r)
        elif r.y<0: w=2*np.pi-vp.diff_angle(vp.vector(1,0,0),r)
        w=w-vp.radians(Marslongperi) # set long. of ascneding node (in this case, w) to be zero at Mars' periapsis longitude
    else: O=O-vp.radians(Marslongperi) # set long. of ascending node to be zero at Mars' periapsis longitude
    return vp.degrees(w)%360,vp.degrees(O)%360,vp.degrees(i)%360,ec,a*6.68459E-12 #return w,O,i,ec,a

def earth2mars(O,w,i,ec,a): 
    # Removes Mars inclination and defines new Keplarian elements with Mars' orbit as the new ecliptic
    earthspace,earthvelo=kep2cart(O,w,i,ec,a,0)
    marsspace=vp.rotate(earthspace,angle=-vp.radians(Marsi),axis=vp.vector(np.cos(vp.radians(MarsO)),np.sin(vp.radians(MarsO)),0))
    marsvelo=vp.rotate(earthvelo,angle=-vp.radians(Marsi),axis=vp.vector(np.cos(vp.radians(MarsO)),np.sin(vp.radians(MarsO)),0))
    return cart2kep(marsspace,marsvelo)


# 2 Burn Method, targets both ascending and descending nodes, then chooses the one with a lower delta-v

def DeltaV2Burn(O,w,i,ec,a):
    DeltaVlist=[]
    for node in [[O,-w],[O-180,180-w]]:
        Mars=kep2cart(0,0,0,Marsec,Marsa,node[0]+180) # starting Mars vectors: true anomaly 180 deg away from each node
        Asteroid=kep2cart(O,w,i,ec,a,node[1]) # at node
        a_transfer=0.5*(Asteroid[0].mag+Mars[0].mag)*6.68459E-12
        ec_transfer=(Asteroid[0].mag-Mars[0].mag)/(Asteroid[0].mag+Mars[0].mag)
        Transfer=kep2cart(0,node[0]+180,0,ec_transfer,a_transfer,0)
        burn_vector=Transfer[1]-Mars[1]
        DeltaV1=np.sqrt(burn_vector.mag2+VMarsesc**2)-VMarspark
        Transfer2=kep2cart(0,node[0]+180,0,ec_transfer,a_transfer,180) # Transfer at asteroid position
        burn_vector2=Asteroid[1]-Transfer2[1]
        DeltaV2=burn_vector2.mag
        DeltaV=DeltaV1+DeltaV2
        TravelTime=0.5*np.sqrt(4*np.pi**2/u*(a_transfer/6.68459E-12)**3)/86400
        DeltaVlist.extend((DeltaV,TravelTime))    
    if DeltaVlist[0]<=DeltaVlist[2]: del DeltaVlist[2:4]
    else: del DeltaVlist[0:2]
    return DeltaVlist


# 3 Burn Method, targets both apoapsis and periapsis, then chooses the one with a lower delta-v

def DeltaV3Burn(O,w,i,ec,a):
    DeltaVlist=[]
    if w<=180: longofperi=vp.degrees(np.arccos(np.cos(vp.radians(w))/np.sqrt(np.cos(vp.radians(w))**2+(np.cos(vp.radians(i))*np.sin(vp.radians(w)))**2)))
    elif w>180: longofperi=-vp.degrees(np.arccos(np.cos(vp.radians(w))/np.sqrt(np.cos(vp.radians(w))**2+(np.cos(vp.radians(i))*np.sin(vp.radians(w)))**2)))
    for apsis in [[O+longofperi,0],[O+longofperi+180,180]]:
        Mars=kep2cart(0,0,0,Marsec,Marsa,apsis[0]+180) # starting Mars vectors: true anomaly 180 deg away from each apsis
        Asteroid=kep2cart(O,w,i,ec,a,apsis[1]) # at apsis
        a_transfer=0.5*(Asteroid[0].mag+Mars[0].mag)*6.68459E-12 # calculate semimajor axis of transfer orbit
        ec_transfer=(Asteroid[0].mag-Mars[0].mag)/(Asteroid[0].mag+Mars[0].mag) # calculate eccentricity of transfer orbit
        Transfer_peri=kep2cart(0,apsis[0]+180,0,ec_transfer,a_transfer,0) # state vectors at periapsis of transfer orbit
        burn_vector=Transfer_peri[1]-Mars[1]
        DeltaV1=np.sqrt(burn_vector.mag2+VMarsesc**2)-VMarspark
        Transfer_90=kep2cart(0,apsis[0]+180,0,ec_transfer,a_transfer,90) # Transfer at plane change position
        Ast_ap_peri_Lat=np.arcsin(np.sin(vp.radians(w+apsis[1]))*np.sin(vp.radians(i))) # ecliptic (mars or otherwise) latitude of apoapsis/periapsis in radians
        Transfer2_90=kep2cart(apsis[0]+180+90,-90,vp.degrees(Ast_ap_peri_Lat),ec_transfer,a_transfer,90) # Transfer at plane change position
        burn_vector2=Transfer_90[1]-Transfer2_90[1]
        DeltaV2=burn_vector2.mag
        Transfer2_ap=kep2cart(apsis[0]+180+90,-90,vp.degrees(Ast_ap_peri_Lat),ec_transfer,a_transfer,180) # Plane changed Transfer at asteroid position
        burn_vector3=Transfer2_ap[1]-Asteroid[1]
        DeltaV3=burn_vector3.mag
        DeltaV=DeltaV1+DeltaV2+DeltaV3
        TravelTime=0.5*np.sqrt(4*np.pi**2/u*(a_transfer/6.68459E-12)**3)/86400
        DeltaVlist.extend((DeltaV,TravelTime))
    if DeltaVlist[0]<=DeltaVlist[2]: del DeltaVlist[2:4]
    else: del DeltaVlist[0:2]
    return DeltaVlist


# Import data from MPCORB.DAT.gz if already in working directory, else downloads the latest verison from the MPC

print('Importing Data...')
# Load datafile, line 44 is data, skip_header=43
try: RAWDATA=np.genfromtxt('MPCORB.DAT.gz',delimiter=(7,6,6,6,10,11,11,11,11,12,14,1,10,6,4,10,6,3,5),skip_header=43,skip_footer=0,usecols=(1,5,6,7,8,10),dtype="float")
except:
    print('Downloading Database...')
    urllib.request.urlretrieve('http://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz','MPCORB.DAT.gz')
    print('Database Downloaded\nImporting Data...')
    RAWDATA=np.genfromtxt('MPCORB.DAT.gz',delimiter=(7,6,6,6,10,11,11,11,11,12,14,1,10,6,4,10,6,3,5,11,1,1,1,1),skip_header=43,skip_footer=0,usecols=(1,5,6,7,8,10),dtype="float")
RAWNAMES=np.genfromtxt('MPCORB.DAT.gz',delimiter=(7,6,6,6,10,11,11,11,11,12,14,1,10,6,4,10,6,3,5,11,1,1,1,1),skip_header=43,skip_footer=0,usecols=(0,11,17,20,21,22,23),dtype="str")


# Filters blank lines from raw imports and prepares the data for processing

EarthDATA=RAWDATA[~np.isnan(RAWDATA).any(axis=1)] # filter out blank lines
NAMES=RAWNAMES[~np.isnan(RAWDATA).any(axis=1)] # filter out blank lines
Hmags=EarthDATA[:,0]
EarthDATA=EarthDATA[:,1:]
EarthDATA=EarthDATA.T
print('Import Complete')


# Define single variable functions for ease of use in a multiprocessing pool

def DVArray2(number):
    array=EarthDATA[:,number]
    array=earth2mars(array[1],array[0],array[2],array[3],array[4])
    return DeltaV2Burn(array[1],array[0],array[2],array[3],array[4])

def DVArray3(number):
    array=EarthDATA[:,number]
    array=earth2mars(array[1],array[0],array[2],array[3],array[4])
    return DeltaV3Burn(array[1],array[0],array[2],array[3],array[4])


# Calculate delta-v's starting at Phobos Orbit

print('Calculating Mars Delta-Vs ('+str(cpus)+' CPUs)...')
with Pool(cpus) as pool:
    DV3Results=pool.map(DVArray3, range(np.shape(EarthDATA)[1]))
    DV2Results=pool.map(DVArray2, range(np.shape(EarthDATA)[1]))
    pool.close()
    pool.join()
    
MarsResults=np.hstack((np.asarray(DV3Results),np.asarray(DV2Results)))

print('Mars Delta-V Calculations Complete')

# Calcaulate Delta-v's for an LEO starting orbit, redefining Mars parameters as Earth parameters to reuse the prior functions.

Marsi=0 # deg
MarsO=174.9 # deg
Marslongperi=102.9 # deg
Marsw=(Marslongperi-MarsO)%360 # deg
Marsa=1 # AU
Marsec=0.0167
MMars=5.972*10**24 # kg
RMars=6.371E6 # Earth Planet Radius (meters)
um=MMars*G
ParkAlt=400000 # Parking Orbit altitude above planet surface (meters) ISS used for Earth
VMarspark=np.sqrt(um*(2/(RMars+ParkAlt)-1/(RMars+ParkAlt))) # Earth centric orbital velocity at parking orbit
VMarsesc=np.sqrt(2*um/(RMars+ParkAlt)) # escape velocity at parking orbit

print('Calculating Earth Delta-Vs ('+str(cpus)+' CPUs)...')
with Pool(cpus) as pool:
    EarthDV3Results=pool.map(DVArray3, range(np.shape(EarthDATA)[1]))
    EarthDV2Results=pool.map(DVArray2, range(np.shape(EarthDATA)[1]))
    pool.close()
    pool.join()

EarthResults=np.hstack((np.asarray(EarthDV3Results),np.asarray(EarthDV2Results)))

print('Earth Delta-V Calculations Complete')

print('Calculating Diameters...') # returning [diameter 0.25, diameter 0.05]
diameters=np.asarray([[np.interp(mag,mag2diameter[:,0],mag2diameter[:,1]),np.interp(mag,mag2diameter[:,0],mag2diameter[:,2])] for mag in Hmags])
print('Diameter Calculations Complete')

print('Creating Output Datafile...')

def best(array): # identify whether 2 or 3 burns provided the lower delta-V
    if array[2]>array[0]:
        return array[0],array[1],3
    else:
        return array[2],array[3],2

def bestearthwrapper(number):
    array=EarthResults[:,number]
    return best(array)

def bestmarswrapper(number):
    array=MarsResults[:,number]
    return best(array)
    
with Pool(cpus) as pool:
    MinResultMars=pool.map(bestmarswrapper, range(np.shape(MarsResults)[1]))
    MinResultEarth=pool.map(bestearthwrapper, range(np.shape(EarthResults)[1]))
    pool.close()
    pool.join()
    
MinResultMars=np.apply_along_axis(best,1,MarsResults)
MinResultEarth=np.apply_along_axis(best,1,EarthResults)

NAMES[:,1][NAMES[:,1]==' ']='-'
NAMES[:,2][NAMES[:,2]=='   ']='---'

EarthDATA=EarthDATA.T

FinalDataTable=np.rec.fromarrays((NAMES[:,0],NAMES[:,1],NAMES[:,2],NAMES[:,3],NAMES[:,4],NAMES[:,5],NAMES[:,6],
                                  Hmags,diameters[:,0]/1000,diameters[:,1]/1000,EarthDATA[:,0],EarthDATA[:,1],EarthDATA[:,2],EarthDATA[:,3],EarthDATA[:,4],
                                  EarthResults[:,0],EarthResults[:,1],EarthResults[:,2],EarthResults[:,3],
                                  MinResultEarth[:,0],MinResultEarth[:,1],MinResultEarth[:,2],
                                  MarsResults[:,0],MarsResults[:,1],MarsResults[:,2],MarsResults[:,3],
                                  MinResultMars[:,0],MinResultMars[:,1],MinResultMars[:,2]))


np.savetxt('DeltaV_Results.txt',FinalDataTable, 
           fmt=('%-7s','%-1s','%-3s','%-1s','%-1s','%-1s','%-1s','%5.2f','%9.3f','%9.3f','%9.5f','%9.5f','%9.5f','%9.7f','%12.7f','%5.0f','%4.0f','%5.0f','%4.0f','%5.0f','%4.0f','%i','%5.0f','%4.0f','%5.0f','%4.0f','%5.0f','%4.0f','%i'),newline='\r\n',
          header='Desn  U Perts orbcode H   diam0.25  diam0.05    Peri.     Node      Incl.   e            a         DV3E TT3E  DV2E TT2E BestE  TTE BME DV3M TT3M  DV2M TT2M BestM TTM BMM')

# np.save('DeltaV_Results.npy', FinalDataTable) #optionally save data as numpy array .npy file

print('Output Datafile Created', '\nAll Processes Complete')