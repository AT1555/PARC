import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,sin,cos,asin,acos,sinh,cosh,asinh,acosh
from functools import lru_cache

def fastcross(a1,a2): #strict 3x3 vector cross product, much faster than np.cross
    return np.array([a1[1]*a2[2]-a1[2]*a2[1],a1[2]*a2[0]-a1[0]*a2[2],a1[0]*a2[1]-a1[1]*a2[0]])

def fastsum(arr):
    return arr[0]+arr[1]+arr[2]

def lambert(r1,r2,t,u=1.327e11): #km,kg,s
    r1,r2=np.asarray(r1),np.asarray(r2)
    c=r2-r1
    absr1,absr2=sqrt(fastsum(r1**2)),sqrt(fastsum(r2**2))
    absc=sqrt(fastsum(c**2))
    s=(absr1+absr2+absc)/2
    ir1,ir2=r1/absr1,r2/absr2
    ih=fastcross(ir1,ir2)
    ih/=sqrt(fastsum(ih**2)) #step not mentioned in paper
    lam=sqrt(1-round(absc/s,4))
    if ih[2]<0:
        lam*=-1
        it1,it2=fastcross(ir1,ih),fastcross(ir2,ih) #wrong in paper
    else: it1,it2=fastcross(ih,ir1),fastcross(ih,ir2) #wrong in paper
    T=sqrt(2*u/s**3)*t
    #sanity check:
    if T<=0: return np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan])
    x,y=findxy(lam,T)
    gamma=sqrt(u*s/2)
    rho=(absr1-absr2)/absc
    sigma=sqrt(1-rho**2)
    vr1=gamma*((lam*y-x)-rho*(lam*y+x))/absr1
    vr2=-gamma*((lam*y-x)+rho*(lam*y+x))/absr2
    vt1,vt2=gamma*sigma*(y+lam*x)/absr1,gamma*sigma*(y+lam*x)/absr2
    v1,v2=vr1*ir1+vt1*it1,vr2*ir2+vt2*it2
    return v1,v2
        
def findxy(lam,T):
    T0=acos(lam)+lam*sqrt(1-lam**2)
    T1=2/3*(1-lam**3)
    if T>=T0: x=(T0/T)**(2/3)-1
    elif T<T1: x=5/2*T1*(T1-T)/T/(1-lam**5)+1
    else: x=(T0/T)**np.log2(T1/T0)-1 #T1<=T<T0
    y=sqrt(1-lam**2*(1-x**2))
    count=0
    while abs((funT(x,y,lam)-T)/T)>0.001:
        x=newton(x,y,lam,T)#Householder(x,y,lam,T)
        y=sqrt(1-lam**2*(1-x**2))
        count+=1
        if count>100: return np.nan, np.nan
    return x,y

def funT(x,y,lam): #0 is M
    if x<1: 
        try: psi=acos(x*y+lam*(1-x**2))
        except ValueError: return np.nan
    #elif x>1: psi=np.arcsinh((y-x*lam)*sqrt(x**2-1))
    elif x>1: psi=acosh(x*y-lam*(x**2-1))
    else: return 2/3*(1-lam**3) # x==1
    return 1/(1-x**2)*((psi+0*np.pi)/sqrt(abs(1-x**2))-x+lam*y)

def newton(x,y,lam,T):
    fx=funT(x,y,lam)-T
    fprimex=(3*(fx+T)*x-2+2*lam**3*x/y)/(1-x**2)
    return x-fx/fprimex

def kep2cart(O,w,i,ec,a,nu,u=1.327e11): # converts orbital elements to cartesian position and velocity vectors, give inputs in deg or AU, returns in km and km/s
    w*=np.pi/180
    i*=np.pi/180
    O*=np.pi/180
    nu*=np.pi/180
    a/=6.68459e-9
    r=a*(1-ec**2)/(1+ec*cos(nu))
    h=sqrt(u*a*(1-ec**2))
    p=a*(1-ec**2)
    #~2x faster with one time calculated trig
    cosO=cos(O)
    coswnu=cos(w+nu)
    sinO=sin(O)
    sinwnu=sin(w+nu)
    cosi=cos(i)
    sini=sin(i)
    sinnu=sin(nu)
    pos=r*np.array([cosO*coswnu-sinO*sinwnu*cosi,
                    sinO*coswnu+cosO*sinwnu*cosi,
                    sini*sinwnu])
    velo=h/r*np.array([(pos[0]*ec/p*sinnu-(cosO*sinwnu+sinO*coswnu*cosi)),
                   (pos[1]*ec/p*sinnu-(sinO*sinwnu-cosO*coswnu*cosi)),
                   (pos[2]*ec/p*sinnu+coswnu*sini)])
    return pos,velo

def cart2kep(r,v,u=1.327e11): # converts cartesian coordinates to orbital elements, all inputs in m and m/s, outputs are in radians and AU, r is the periapsis position vector, v is the velocity vector at periapsis
    r,v=np.asarray(r),np.asarray(v)
    if np.isnan(v[0]): return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan 
    h=fastcross(r,v) # orbital momentum vector, perpendicular to orbital plane
    if h[0]==0 and h[1]==0: #zero inclination case, add slight perturbation to make math work
        v+=np.ones(3)*sqrt(fastsum(v**2))/1e7
        h=fastcross(r,v)
    evec=fastcross(v,h)/u-r/sqrt(fastsum(r**2)) #eccentricity vector
    ec=sqrt(fastsum(evec**2))
    n=fastcross([0,0,1],h) #vector to ascending node
    i=acos(h[2]/sqrt(fastsum(h**2)))
    if fastsum(r*v)>=0: nu=acos(round(fastsum(evec*r)/sqrt(fastsum(r**2)*fastsum(evec**2)),5))
    else: nu=2*np.pi-acos(round(fastsum(evec*r)/sqrt(fastsum(r**2)*fastsum(evec**2)),5))
    if n[1]>=0: O=acos(round(n[0]/sqrt(fastsum(n**2)),5))
    else: O=2*np.pi-acos(round(n[0]/sqrt(fastsum(n**2)),5))
    if evec[2]>=0: w=acos(round(fastsum(n*evec)/sqrt(fastsum(evec**2)*fastsum(n**2)),5))
    else: w=2*np.pi-acos(round(fastsum(n*evec)/sqrt(fastsum(evec**2)*fastsum(n**2)),5))
    a=1/((2/sqrt(fastsum(r**2)))-(fastsum(v**2)/u))
    return O*180/np.pi,w*180/np.pi,i*180/np.pi,ec,a*6.68459e-9,nu*180/np.pi #return O,w,i,ec,a,nu

def plotorb(O,w,i,ec,a,nu,label=''):
    plt.scatter(0,0,marker='*',c='gold',s=100)
    if ec<1: 
        points=np.array([kep2cart(O,w,i,ec,a,nu)[0] for nu in np.linspace(0,360,num=360)])
        plt.plot(points[:,0],points[:,1],label=label)
        return
    else: return
    
def gen_ephemeris(epoch,M,O,w,inc,e,a,date,u=1.327e11):
    M=(date-epoch).item().days*24*3600/sqrt((a*1.496e8)**3/u)*180/np.pi+M
    E=M*np.pi/180 #inital guess
    count=0
    while np.abs(E-e*sin(E)-M*np.pi/180)>1e-3:
        E-=(E-e*sin(E)-M*np.pi/180)/(1-e*cos(E))
        count+=1
        if count>100: return np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan])
    E=E%(2*np.pi)
    #print(E*180/np.pi)
    nu=acos((cos(E)-e)/(1-e*cos(E)))*180/np.pi
    if E>np.pi: nu=360-nu
    return kep2cart(O,w,inc,e,a,nu)

@lru_cache(maxsize=2000)
def DeltaV2Burn(epoch,M,node,peri,inc,e,a,date,tof,planet='mars',plot=0):
    if planet=='mars':
        vpark=2.1376
        vesc=3.023
        localephem=gen_ephemeris(np.datetime64('2000-01-01'),355.45332,49.57854,336.04084,1.85061,0.09341233,1.52366231,date)
    else: 
        vpark=7.6723634
        vesc=10.8503604
        localephem=gen_ephemeris(np.datetime64('2000-01-01'),100.46435,-11.26064,102.94719,0.00005,0.01671022,1.00000011,date)
    targetephem=gen_ephemeris(epoch,M,node,peri,inc,e,a,date+tof)
    v1,v2=lambert(localephem[0],targetephem[0],tof*24*3600)
    vinfinity=sqrt(fastsum((localephem[1]-v1)**2))
    DeltaV1=sqrt(vinfinity**2+vesc**2)-vpark
    DeltaV2=sqrt(fastsum((targetephem[1]-v2)**2))
    if plot==1:
        print(f"{DeltaV1+DeltaV2:.3f} km/s = {DeltaV1:.3f} km/s + {DeltaV2:.3f} km/s")
        print(f"Launch date: {date}, Arrive in {tof} days")
        plt.scatter(localephem[0][0],localephem[0][1])
        plt.scatter(targetephem[0][0],targetephem[0][1])
        plt.scatter(0,0,c='gold',marker='*')
        plotorb(*cart2kep(*localephem))
        plotorb(*cart2kep(*targetephem))
        plotorb(*cart2kep(localephem[0],v1))
        plt.axis('equal')
        plt.show()
    return DeltaV1+DeltaV2

@lru_cache(maxsize=2000)
def DeltaV3Burn(epoch,M,node,peri,inc,e,a,date,tof,planet='mars',u=1.327e11,plot=0): 
    if planet=='mars':
        vpark=2.1376
        vesc=3.023
        localephem=gen_ephemeris(np.datetime64('2000-01-01'),355.45332,49.57854,336.04084,1.85061,0.09341233,1.52366231,date)
    else: 
        vpark=7.6723634
        vesc=10.8503604
        localephem=gen_ephemeris(np.datetime64('2000-01-01'),100.46435,-11.26064,102.94719,0.00005,0.01671022,1.00000011,date)
    targetephem=gen_ephemeris(epoch,M,node,peri,inc,e,a,date+tof)
    #set semimajor axis of transfer orbit, then execute broken plane maneuver when halfway in angle in plane of starting orbit between origin and target
    #rotate target vector into the plane of the starting orbit
    planenormal=fastcross(localephem[0],localephem[1])/sqrt(fastsum(localephem[0]**2)*fastsum(localephem[1]**2))
    targetinplane=targetephem[0]-fastsum(targetephem[0]*planenormal)*planenormal
    targetinplane*=sqrt(fastsum(targetephem[0]**2)/fastsum(targetinplane**2))
    #Calulate first burn
    v1,v2=lambert(localephem[0],targetinplane,tof*24*3600)
    vinfinity=sqrt(fastsum((localephem[1]-v1)**2))
    DeltaV1=sqrt(vinfinity**2+vesc**2)-vpark
    transorb=cart2kep(localephem[0],v1)
    if np.isnan(transorb[0]): return np.nan
    if transorb[3]>1: return np.nan #discard hyperbolic case, not worth it
    #when halfway in angle to target rotated into plane: execute broken plane manuever
    try: fullangle=acos(round(fastsum(localephem[0]*targetinplane)/sqrt(fastsum(localephem[0]**2)*fastsum(targetinplane**2)),5)) #radians
    except ValueError: return np.nan
    if fastcross(localephem[0],targetinplane)[2]<0: fullangle=2*np.pi-fullangle 
    halfway=kep2cart(*transorb[:5],transorb[5]+fullangle*180/np.pi/2)
    #calculate remaining tof
    Estart=acos((transorb[3]+cos(transorb[5]*np.pi/180))/(1+transorb[3]*cos(transorb[5]*np.pi/180))) #radians
    Mstart=Estart-transorb[3]*sin(Estart) #radians
    Ehalf=acos((transorb[3]+cos(transorb[5]*np.pi/180+fullangle/2))/(1+transorb[3]*cos(transorb[5]*np.pi/180+fullangle/2))) #radians
    Mhalf=Ehalf-transorb[3]*sin(Ehalf) #radians
    meanmotion=sqrt(u/(transorb[4]*1.496e8)**3) #radians/sec
    deltat=np.abs(Mhalf-Mstart)%(2*np.pi)/meanmotion #sec
    #calculate broken plane maneuver
    v3,v4=lambert(halfway[0],targetephem[0],tof*24*3600-deltat)
    DeltaV2=sqrt(fastsum((halfway[1]-v3)**2))
    DeltaV3=sqrt(fastsum((targetephem[1]-v4)**2))
    if plot==1:
        print(f"{DeltaV1+DeltaV2+DeltaV3:.3f} km/s = {DeltaV1:.3f} km/s + {DeltaV2:.3f} km/s + {DeltaV3:.3f} km/s")
        print(f"Launch date: {date}, Arrive in {tof} days")
        #print(fullangle*180/np.pi)
        plt.scatter(localephem[0][0],localephem[0][1])
        plt.scatter(targetephem[0][0],targetephem[0][1])
        plt.scatter(halfway[0][0],halfway[0][1])
        plt.scatter(0,0,c='gold',marker='*')
        plotorb(*cart2kep(*localephem))
        plotorb(*cart2kep(*targetephem))
        plotorb(*transorb)
        plt.axis('equal')
        plt.show()
    return DeltaV1+DeltaV2+DeltaV3

def rotatevector(a,b,theta): #rotate a theta radians around b 
    a=np.asarray(a).astype(float)
    b=np.asarray(b).astype(float)
    b/=sqrt(fastsum(b**2))
    projab=fastsum(a*b)*b
    perpproj=a-projab
    crossvec=fastcross(b,perpproj)
    return projab+perpproj*cos(theta)+sqrt(fastsum(perpproj**2))*crossvec/sqrt(fastsum(crossvec**2))*sin(theta)

def earth2mars(O,w,i,ec,a): #rotate orbit from earth-ecliptic orbital elements to mars-ecliptic orbital elements
    target=kep2cart(O,w,i,ec,a,0)
    marsnode=kep2cart(49.57854,336.04084-49.57854,1.85061,0.09341233,1.52366231,0-(336.04084-49.57854))
    targetmarsframe=rotatevector(target[0],marsnode[0],-1.85061*np.pi/180)
    targetvmarsframe=rotatevector(target[1],marsnode[0],-1.85061*np.pi/180)
    return cart2kep(targetmarsframe,targetvmarsframe)

# 2 Burn Method, targets both ascending and descending nodes, then chooses the one with a lower delta-v

def DeltaV2BurnTest(O,w,i,ec,a,planet='mars',u=1.327e11):
    DeltaVlist=[]
    if planet=='mars':
        vpark=2.1376
        vesc=3.023
        #rotate asteroid orbit into Mars frame of reference
        targetorb=earth2mars(O,w,i,ec,a)
    else: 
        vpark=7.672363406552731
        vesc=10.850360385001911
        targetorb=np.array([O,w,i,ec,a])
    for node in [0,180]:
        targetephem=kep2cart(*targetorb[:5],node-targetorb[1])
        if planet=='mars': localephem=kep2cart(49.57854,336.04084-49.57854,0,0.09341233,1.52366231,180-node+targetorb[0]-336.04084+1e-2) #+1e-2 is to force lambert solver to work nicely
        else: localephem=kep2cart(-11.26064,102.94719,0.00005,0.01671022,1.00000011,180-node+targetorb[0]-(-11.26064+102.94719)+1e-2) #+1e-2 is to force lambert solver to work nicely
        a_transfer=0.5*(sqrt(fastsum(localephem[0]**2))+sqrt(fastsum(targetephem[0]**2)))
        tof=np.pi*sqrt(a_transfer**3/u)
        v1,v2=lambert(localephem[0],targetephem[0],tof)
        vinfinity=sqrt(fastsum((localephem[1]-v1)**2))
        DeltaV1=sqrt(vinfinity**2+vesc**2)-vpark
        DeltaV2=sqrt(fastsum((targetephem[1]-v2)**2))
        DeltaVlist.append(DeltaV1+DeltaV2) 
    return min(DeltaVlist)

# 3 Burn Method, targets both apoapsis and periapsis, then chooses the one with a lower delta-v

def DeltaV3BurnTest(O,w,i,ec,a,planet='mars',u=1.327e11):
    if planet=='mars':
        vpark=2.1376
        vesc=3.023
        localperi=kep2cart(49.57854,336.04084-49.57854,1.85061,0.09341233,1.52366231,0)
    else: 
        vpark=7.672363406552731
        vesc=10.850360385001911
        localperi=kep2cart(-11.26064,102.94719,0.00005,0.01671022,1.00000011,0)
    normalvec=fastcross(localperi[0],localperi[1])/np.sqrt(fastsum(localperi[0]**2)*fastsum(localperi[1]**2))
    peri=kep2cart(O,w,i,ec,a,0)
    peri_local_plane=peri[0]-fastsum(peri[0]*normalvec)*normalvec
    localnu_offset=acos(fastsum(localperi[0]*peri_local_plane)/np.sqrt(fastsum(localperi[0]**2)*fastsum(peri_local_plane**2)))*180/np.pi
    if fastcross(localperi[0],peri_local_plane)[2]<0: localnu_offset=360-localnu_offset
    DeltaVlist=[]
    for apsis in [0,180]:
        targetephem=kep2cart(O,w,i,ec,a,apsis)
        if planet=='mars': localephem=kep2cart(49.57854,336.04084-49.57854,1.85061,0.09341233,1.52366231,180-apsis+localnu_offset+1e-2) #+1e-2 is to force lambert solver to work nicely
        else: localephem=kep2cart(-11.26064,102.94719,0.00005,0.01671022,1.00000011,180-apsis+localnu_offset+1e-2) #+1e-2 is to force lambert solver to work nicely
        a_transfer=0.5*(sqrt(fastsum(localephem[0]**2))+sqrt(fastsum(targetephem[0]**2)))
        tof=np.pi*sqrt(a_transfer**3/u)
        targetinplane=targetephem[0]-fastsum(targetephem[0]*normalvec)*normalvec
        targetinplane*=sqrt(fastsum(targetephem[0]**2)/fastsum(targetinplane**2))
        v1,v2=lambert(localephem[0],targetinplane,tof)
        vinfinity=sqrt(fastsum((localephem[1]-v1)**2))
        DeltaV1=sqrt(vinfinity**2+vesc**2)-vpark
        transorb=cart2kep(localephem[0],v1)
        if transorb[3]>1: return np.nan #discard hyperbolic case, not worth it
        #when halfway in angle to target rotated into plane: execute broken plane manuever
        halfway=kep2cart(*transorb[:5],transorb[5]+90)
        #calculate remaining tof
        Estart=acos((transorb[3]+cos(transorb[5]*np.pi/180))/(1+transorb[3]*cos(transorb[5]*np.pi/180))) #radians
        Mstart=Estart-transorb[3]*sin(Estart) #radians
        Ehalf=acos((transorb[3]+cos(transorb[5]*np.pi/180+np.pi/2))/(1+transorb[3]*cos(transorb[5]*np.pi/180+np.pi/2))) #radians
        Mhalf=Ehalf-transorb[3]*sin(Ehalf) #radians
        meanmotion=sqrt(u/(transorb[4]*1.496e8)**3) #radians/sec
        deltat=np.abs(Mhalf-Mstart)%(2*np.pi)/meanmotion #sec
        #calculate broken plane maneuver
        v3,v4=lambert(halfway[0],targetephem[0],tof-deltat)
        DeltaV2=sqrt(fastsum((halfway[1]-v3)**2))
        DeltaV3=sqrt(fastsum((targetephem[1]-v4)**2))
        DeltaVlist.append(DeltaV1+DeltaV2+DeltaV3)
    return min(DeltaVlist)
