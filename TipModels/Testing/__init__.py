from common.baseclasses import AWA
from common.log import Logger
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import solve
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat

layers=mat.LayeredMedia((mat.SiO2_300nm,50e-7),exit=mat.Si)

def TestDipoleLimit(freqs=linspace(900,1300,50),material=layers,\
                    a=30,zmin=1,Nqs=72,by_hand=False):
    #zmin=1 is just for didacticism - here the greatest mathematical differences are revealed
    
    global rps,GMat,LambdaMat,kernel,Lambda0Vec,new_dipole_moments

    #Put some settings and load data for replacement
    tip.LRM.quadrature_params['xWarp']=False
    _=tip.LRM(1000,rp=material.reflection_p,demodulate=False,normalize_to=None,\
              Nzs=1,zmin=zmin,a=a,Nqs=Nqs)
    tip.LRM.load_params['reload_model']=False
    tip.LRM.load_params['comsol_lambda0']=False
    
    #Get ss & qs and build new lambdas
    wqs=tip.LRM.wqs
    old_Lambda=tip.LRM.Lambda
    old_Lambda0=tip.LRM.Lambda0
    ss_grid,qs_grid=old_Lambda.axis_grids
    ss=old_Lambda0.axes[0]
    
    new_Lambda0=AWA(-ss*exp(-ss))
    new_Lambda0.adopt_axes(old_Lambda0)
    tip.LRM.Lambda0=new_Lambda0
    
    new_Lambda=AWA(-ss_grid*exp(-(ss_grid+qs_grid)))
    new_Lambda.adopt_axes(old_Lambda)
    tip.LRM.Lambda=new_Lambda
    
    tip.LRM.geometric_params['acceptance_angle']=60
    old_dipole_moments=tip.LRM.dipole_moments[60]
    new_dipole_moments=AWA(exp(-ss))
    new_dipole_moments.adopt_axes(old_dipole_moments)
    tip.LRM.dipole_moments[60]=new_dipole_moments
    
    #Simulate side-by-side
    ##Note this test is only valid for zmin>=a in dipole model, since those are the conditions
    #for which the dipole equivalence was derived
    slice_dp=tip.DipoleModel(freqs,rp=material.reflection_p,demodulate=False,normalize_to=None,\
                             Nzs=1,zmin=a+zmin,a=a)
    
    tip.LRM.quadrature_params['q_correction']= False
    slice_quasi_dp_1=tip.LRM(freqs,rp=material.reflection_p,demodulate=False,normalize_to=None,\
                             Nzs=1,zmin=zmin,a=a,Nqs=Nqs)['signals']
    
    tip.LRM.quadrature_params['q_correction']= True
    slice_quasi_dp_2=tip.LRM(freqs,rp=material.reflection_p,demodulate=False,normalize_to=None,\
                             Nzs=1,zmin=zmin,a=a,Nqs=Nqs)['signals']
    
    figure()
    abs(slice_dp).plot(label='Actual dipole')
    abs(slice_quasi_dp_1).plot(label='Quasi dipole')
    abs(slice_quasi_dp_2).plot(label='Quasi dipole q-corrected')
    
    #Simulate by hand with scattering method#
    if by_hand:
        
        zmin=zmin/float(a)
        I=matrix(eye(len(ss)))
        WMat=matrix(diag(wqs))
        LambdaMat=matrix(new_Lambda)
        Lambda0Vec=matrix(new_Lambda0).T
        
        signals1=[]
        signals2=[]
        Logger.write('Computing by hand...')
        rps=material.reflection_p(freqs,ss/float(a*1e-7))
        for i,freq in enumerate(freqs):
            rp=rps.cslice[freq]
            GMat=matrix(diag(-ss*exp(-2*zmin*ss)*rp))
            kernel=LambdaMat*WMat*GMat
            soln=solve((I-kernel),Lambda0Vec)
            signals1.append(sum(new_dipole_moments*wqs*array(GMat*soln).squeeze()))
            signals2.append(sum(new_dipole_moments*array(GMat*soln).squeeze()))
            Logger.write('\tProgress: %i%%'%(i/float(len(freqs))*100))
        
        signals1=AWA(signals1,axes=[freqs],axis_names=['Frequency'])
        signals2=AWA(signals2,axes=[freqs],axis_names=['Frequency'])
        abs(signals1).plot(label='By Hand - with weights')
        abs(signals2).plot(label='By Hand - without weights')
    
    for l in gca().lines: l.set_ydata(l.get_ydata()/l.get_ydata().max())
    ylim(0,1.2)
    legend(loc='best')
    