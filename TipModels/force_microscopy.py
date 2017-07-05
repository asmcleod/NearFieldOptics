import numpy
from common import misc
from common.log import Logger
from scipy.integrate import simps,trapz
from common.baseclasses import ArrayWithAxes as AWA
from common.numerics import Spectrum
from common import numerical_recipes as numrec
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat

def compute_bulk_force(rp=mat.SiO2_Bulk.reflection_p,freq=1160,\
                       zmins=numpy.logspace(0,3,100),a=20,Nqs=72,Nzs=500,zmax=1000,\
                       verbose=True):
    
    tip.verbose=False
    alpha=numpy.tan(20/180.*numpy.pi)
    bulk_rp=rp(freq,q=1/(a*1e-7)).squeeze()
    zs=numpy.logspace(-1,numpy.log(zmax)/numpy.log(10),Nzs)*1e-7 #in cm
    dzs=list(numpy.diff(zs)); dzs.append(dzs[-1]); dzs=numpy.array(dzs)
    
    zs1=numpy.reshape(zs,(len(zs),1))
    zs2=numpy.reshape(zs,(1,len(zs)))
    dzs1=numpy.reshape(dzs,(len(zs),1))
    dzs2=numpy.reshape(dzs,(1,len(zs)))
    
    global charges,DC_forces,AC_forces,s3s
    charges=[]
    DC_forces=[]
    AC_forces=[]
    s3s=[]
    
    if verbose: Logger.write('Computing bulk force for %i zmin values...'%len(zmins))
    for i,zmin in enumerate(zmins):
        if verbose: Logger.write('\tzmin=%1.2f nm, Progress: %i%%'%(zmin,i/float(len(zmins))*100))
        s3=tip.SSEQModel2(freq,a=a,zmin=zmin,rp=rp,\
                              normalize_to=None,Nqs=72,\
                              amplitude=0,Nzs=1,demodulate=False)
        s3s.append(s3)
        
        charge=tip.SSEQModel2.get_charge_distribution(zs)
        charges.append(charge)
        
        charge1=numpy.reshape(charge,(len(zs),1))
        
        charge2=numpy.reshape(charge,(1,len(zs)))
        #This distance kernel for interacting monopole rings is a conservative underestimate
        distance_kernel=(zs1+zs2+2*zmin*1e-7)/((zs1+zs2+2*zmin*1e-7)**2+((zs1+zs2)*alpha)**2)**(3/2.)
        #distance_kernel=(zs1+zs2+2*zmin)/((zs1+zs2+2*zmin)**2+(zs2*alpha)**2)**(3/2.)
        #distance_kernel=1/(zs1+zs2+2*zmin*1e-7)**2
        field=-bulk_rp*charge2*distance_kernel
        
        #Compute force, use cm as units of distance
        DC_force=simps(simps(\
                             (charge1*numpy.conj(field)+numpy.conj(charge1)*field)/4.,\
                             x=zs,axis=0),x=zs,axis=0)
        #Let AC part remain complex, the phase describes phase of the 2*omega part
        AC_force=simps(simps(\
                             charge1*field,\
                             x=zs,axis=0),x=zs,axis=0)
        
        DC_forces.append(DC_force)
        AC_forces.append(AC_force)
        
    charges=AWA(charges,axes=[zmins],axis_names=['$z_{min}$'])
    AC_forces=AWA(AC_forces,axes=[zmins],axis_names=['$z_{min}$'])
    DC_forces=AWA(DC_forces,axes=[zmins],axis_names=['$z_{min}$'])
    s3s=AWA(s3s,axes=[zmins],axis_names=['$z_{min}$'])
    
    return {'charges':charges,\
            'AC_forces':AC_forces,\
            'DC_forces':DC_forces,\
            's3s':s3s}
    
def average_force_spectrum(freqs=numpy.linspace(500,1800,20),rp=mat.SiO2_Bulk.reflection_p,\
                           a=20,zmin=1,A=50,Nts=100,Nqs=72,Nzs=500,zmax=1000,\
                           verbose=True):
    
    ts=numpy.logspace(-3,0,Nts) #Use a log scale in time, best resolving the "closest" times
    zmins=zmin+A/2.*(1-numpy.cos(numpy.pi*ts))
    c=3e8 #in m/s
    NA=1.5 #numerical aperture of mirror
    P=1e-3 #laser power in Watts
    enhancement=100 #Lightning rod effect enhancement
    
    global forces, d
    forces=[]
    if verbose: Logger.write('Computing bulk force for %i frequency values...'%len(freqs))
    for i,freq in enumerate(freqs):
        if verbose: Logger.write('\tfreq=%1.2f cm-1, Progress: %i%%'%(freq,i/float(len(freqs))*100))
        
        wl=1/float(freq)*1e-2 #wavelength from cm to m
        w=.44*wl/(2.*NA) #in m
        E0=2/w*numpy.sqrt(P/c)
        lambda0=enhancement*3/2.*E0*(a*1e-9) #in units of sqrt(N)
        
        d=compute_bulk_force(rp=mat.SiO2_Bulk.reflection_p,freq=freq,zmins=zmins,a=a,Nqs=Nqs,Nzs=Nzs,zmax=zmax,\
                             verbose=False)
        avg_DC_force=simps(d['DC_forces'],x=ts)
        force=(lambda0*avg_DC_force)**2
        if verbose: Logger.write('force: %1.2f N'%force)
        forces.append(force)
        
    return AWA(forces,axes=[freqs],axis_names=['Frequency'])
        