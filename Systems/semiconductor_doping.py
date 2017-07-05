import numpy
from matplotlib.pyplot import *
from common.log import Logger
from common.baseclasses import AWA
from NearFieldOptics import Materials as mat
from NearFieldOptics import TipModels as tip

e_dopings=numpy.logspace(16, 21, 100)
h_dopings=0

def signal_vs_doping(e_dopings=e_dopings,h_dopings=h_dopings,wavelength=10.7,\
                     norm_doping=1e15,a=25,amplitude=40,Nzs=20,Nqs=72,\
                     model=tip.LRM):
    
    N=len(e_dopings)
    if not hasattr(h_dopings,'__len__'): h_dopings=[h_dopings]*N
    
    freq=1e4/float(wavelength)
    load_freq=1100
    
    Logger.write('Getting normalization signal at doping n=%1.1g...'%norm_doping)
    Si_norm=mat.DopedSilicon(ne=norm_doping,\
                             nh=0)
    tip.verbose=False
    if model is tip.LRM:
        norm_signal=tip.LRM(freq,rp=Si_norm.reflection_p,zmin=.1,amplitude=amplitude,a=a,\
                            normalize_to=None,normalize_at=freq,Nzs=Nzs,Nqs=Nqs,\
                            load_freq=load_freq*25/1e7,load_Nzs=144,load_Nqs=144,taper_angle=20,geometry='FiniteCone')
    
    else:
        norm_signal=tip.DipoleModel(freq,rp=Si_norm.reflection_p,zmin=.7*a,amplitude=amplitude,a=a,\
                                    normalize_to=None,normalize_at=freq,Nzs=20,Nqs=500)
    
    tip.LRM.load_params['reload_model']=False
    
    global rps,s2s,s3s
    rps=[]
    s2s=[]
    s3s=[]
    
    for i,dopings in enumerate(zip(e_dopings,h_dopings)):
        Logger.write('\tPROGRESS: %1.2f%% complete...'%(i/float(N)*100))
        
        e_doping,h_doping=dopings
        mat.Si_Doped.set_doping(e_doping,h_doping)
        rps.append(mat.Si_Doped.reflection_p(freq,q=1e7/float(a)))
        
        if model is tip.LRM:
            signal=tip.LRM(freq,rp=mat.Si_Doped.reflection_p,zmin=.1,amplitude=amplitude,a=a,\
                           normalize_to=None,normalize_at=freq,Nzs=Nzs,Nqs=Nqs,\
                           load_freq=load_freq*25/1e7,load_Nzs=144,taper_angle=20,geometry='FiniteCone')
        else:
            signal=tip.DipoleModel(freq,rp=mat.Si_Doped.reflection_p,zmin=.7*a,amplitude=amplitude,a=a,\
                                    normalize_to=None,normalize_at=freq,Nzs=20,Nqs=500)
        
        s2s.append((signal['signal_2']/norm_signal['signal_2']).squeeze())
        s3s.append((signal['signal_3']/norm_signal['signal_3']).squeeze())
        
    tip.LRM.load_params['reload_model']=True
    tip.verbose=True
        
    if numpy.max(e_dopings)-numpy.min(e_dopings) >\
       numpy.max(h_dopings)-numpy.min(h_dopings):
        doping_axis=e_dopings
    else: doping_axis=h_dopings
        
    rps=AWA(rps,axes=[doping_axis],axis_names=['Doping'])
    s2s=AWA(s2s,axes=[doping_axis],axis_names=['Doping'])
    s3s=AWA(s3s,axes=[doping_axis],axis_names=['Doping'])
    
    
    figure()
    (10*rps.imag).plot(plotter=semilogx,ls='--',color='r',lw=2)
    numpy.abs(s3s).plot(color='b',lw=2)
    ylabel(r'$S_3/S_{3\,%1.1g}$'%norm_doping,\
           fontsize=25,color='b')
    gca().spines['left'].set_color('b')
    yticks(color='b')
    
    twinx()
    plot(doping_axis,numpy.unwrap(numpy.angle(s3s))*(180/numpy.pi),color='r',lw=2)
    gca().spines['right'].set_color('r')
    yticks(color='r')
    
    ylabel(r'$\phi_3-\phi_{3\,%1.1g}\,[\mathrm{deg.}]$'%norm_doping,\
           fontsize=25,color='r',rotation=270)
    
    
    figure()
    (10*rps.imag).plot(plotter=semilogx,ls='--',color='r',lw=2)
    numpy.abs(s2s).plot(color='b',lw=2)
    ylabel(r'$S_2/S_{2\,%1.1g}$'%norm_doping,\
           fontsize=25,color='b')
    gca().spines['left'].set_color('b')
    yticks(color='b')
    
    twinx()
    plot(doping_axis,numpy.unwrap(numpy.angle(s2s))*(180/numpy.pi),color='r',lw=2)
    gca().spines['right'].set_color('r')
    yticks(color='r')
    
    ylabel(r'$\phi_2-\phi_{2\,%1.1g}\,[\mathrm{deg.}]$'%norm_doping,\
           fontsize=25,color='r',rotation=270)
    
    
    return {'s2':s2s,'s3':s3s,'rps':rps}
        