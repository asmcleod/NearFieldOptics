import os
from numpy import *
from matplotlib.pyplot import *
from common.misc import sort_by
from common.plotting import grid_axes
from common.numerical_recipes import smooth
import cPickle
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat


def PlotBulkSpectroscopy():
    
    ##Open data##
    data_path='/Users/alexmcleod/tools/python/NearFieldOptics/Data/Amplitude_Zmin_Effects'
    sic_data_path=os.path.join(data_path,'SiC_Amplitude=40nm_smoothed_curves.pickle')
    sio2_data_path=os.path.join(data_path,'Experiment1','SiO2_Amplitude=60nm_smoothed_curves.pickle')
    d_sio2=cPickle.load(open(sio2_data_path))
    d_sic=cPickle.load(open(sic_data_path))
    
    ##Delineate data limits##
    sio2_phase_xlims=[1000,1175]
    sio2_signal_xlims=[1000,1200]
    sic_phase_xlims=[700,955]
    sic_signal_xlims=[700,975]
    
    ##Plot properties##
    f=figure()
    s2_color=(.3,0,.7)
    s3_color=(1,0,0)
    marker_kwargs=dict(marker='o',markersize=5,ls='')
    line_kwargs=dict(lw=2)
    
    ##Plot SiO2 signal
    s2_smoothed_signal=d_sio2['smoothed_amplitude']['S_2']
    abs(s2_smoothed_signal.cslice[sio2_signal_xlims[0]:\
                                  sio2_signal_xlims[1]]).plot(color=s2_color,**line_kwargs)
    s3_smoothed_signal=d_sio2['smoothed_amplitude']['S_3']
    abs(s3_smoothed_signal.cslice[sio2_signal_xlims[0]:\
                                  sio2_signal_xlims[1]]).plot(color=s3_color,**line_kwargs)
    s2_raw_signal=d_sio2['raw_amplitude']['S_2']
    abs(s2_raw_signal.cslice[sio2_signal_xlims[0]:\
                             sio2_signal_xlims[1]]).plot(color=s2_color,**marker_kwargs)
    s3_raw_signal=d_sio2['raw_amplitude']['S_3']
    abs(s3_raw_signal.cslice[sio2_signal_xlims[0]:\
                             sio2_signal_xlims[1]]).plot(color=s3_color,**marker_kwargs)
                             
    
    ##Plot SiC signal##
    s2_smoothed_signal=d_sic['smoothed_amplitude']['S_2']
    abs(s2_smoothed_signal.cslice[sic_signal_xlims[0]:\
                                  sic_signal_xlims[1]]).plot(color=s2_color,**line_kwargs)
    s3_smoothed_signal=d_sic['smoothed_amplitude']['S_3']
    abs(s3_smoothed_signal.cslice[sic_signal_xlims[0]:\
                                  sic_signal_xlims[1]]).plot(color=s3_color,**line_kwargs)
    
    s2_raw_signal=d_sic['raw_amplitude']['S_2']
    abs(s2_raw_signal.cslice[sic_signal_xlims[0]:\
                             sic_signal_xlims[1]]).plot(color=s2_color,**marker_kwargs)
    s3_raw_signal=d_sic['raw_amplitude']['S_3']
    abs(s3_raw_signal.cslice[sic_signal_xlims[0]:\
                             sic_signal_xlims[1]]).plot(color=s3_color,**marker_kwargs)
    ylim(0,5.5)
    
    ##Now plot phases##
    twinx()
    line_kwargs=dict(lw=2,ls='--')
    
    s2_smoothed_phase=d_sio2['smoothed_phase']['S_2']+2*pi
    (s2_smoothed_phase.cslice[sio2_phase_xlims[0]:\
                              sio2_phase_xlims[1]]/(2*pi)).plot(color=s2_color,**line_kwargs)
    s3_smoothed_phase=d_sio2['smoothed_phase']['S_3']-2*pi
    (s3_smoothed_phase.cslice[sio2_phase_xlims[0]:\
                              sio2_phase_xlims[1]]/(2*pi)).plot(color=s3_color,**line_kwargs)
    
    s2_smoothed_phase=d_sic['smoothed_phase']['S_2']+4*pi
    (s2_smoothed_phase.cslice[sic_phase_xlims[0]:\
                              sic_phase_xlims[1]]/(2*pi)).plot(color=s2_color,**line_kwargs)
    s3_smoothed_phase=d_sic['smoothed_phase']['S_3']+6*pi
    (s3_smoothed_phase.cslice[sic_phase_xlims[0]:\
                              sic_phase_xlims[1]]/(2*pi)).plot(color=s3_color,**line_kwargs)
    ylim(-1,1.75)
                              
    ##Now add legend##
    lines=f.axes[0].lines[:2]+f.axes[1].lines[:2]
    labels=['$S_2$','$S_3$','$\Phi_2$','$\Phi_3$']
    sca(f.axes[1])
    gca().set_yticks((-.5,0,.5,1,1.5))
    yticks(fontsize=16)
    ylabel('$\Phi-\Phi_{Au}\,[2\pi]$',fontsize=25,rotation=270)
    l=legend(lines,labels,shadow=True,fancybox=True,loc='best')
    for t in l.texts: t.set_fontsize(16)
    
    sca(f.axes[0])
    gca().set_yticks((1,2,3,4,5))
    xticks(fontsize=16)
    yticks(fontsize=16)
    xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    ylabel('$|S|/|S_{Au}|$',fontsize=25)
    axhline(0,ls='--',color='k',lw=1.5)
    
    return f

def PlotBulkModeling():
    
    ##Delineate data limits##
    sio2_phase_xlims=[1000,1200]
    sio2_signal_xlims=[1000,1200]
    sic_phase_xlims=[700,1000]
    sic_signal_xlims=[700,1000]
    
    
    ##Get all data##
    freqs=linspace(sic_signal_xlims[0],\
                       sic_signal_xlims[1],200)
    rp_sic=mat.SiC_4H.reflection_p(freqs,angle=60)
    rp_au=mat.Au.reflection_p(freqs,angle=60)
    ff_sic=(1+rp_sic)**2/(1+rp_au)**2
    s3_sic_dp=tip.DipoleModel(freqs,rp=mat.SiC_4H.reflection_p,\
                              amplitude=40,Nzs=30,harmonic=3,\
                              normalize_to=mat.Au.reflection_p,zmin=23.5,a=30)
    s2_sic_dp=tip.DipoleModel(freqs,rp=mat.SiC_4H.reflection_p,\
                              amplitude=40,Nzs=30,harmonic=2,\
                              normalize_to=mat.Au.reflection_p,zmin=23.5,a=30)
    freqs=linspace(sio2_signal_xlims[0],\
                       sio2_signal_xlims[1],200)
    rp_sio2=mat.SiO2_300nm.reflection_p(freqs,angle=60)
    rp_au=mat.Au.reflection_p(freqs,angle=60)
    ff_sio2=(1+rp_sio2)**2/(1+rp_au)**2
    s3_sio2_dp=tip.DipoleModel(freqs,rp=mat.SiO2_300nm.reflection_p,\
                              amplitude=40,Nzs=30,harmonic=3,\
                              normalize_to=mat.Au.reflection_p,zmin=23.5,a=30)
    s2_sio2_dp=tip.DipoleModel(freqs,rp=mat.SiO2_300nm.reflection_p,\
                              amplitude=40,Nzs=30,harmonic=2,\
                              normalize_to=mat.Au.reflection_p,zmin=23.5,a=30)
    
    ##Plot properties##
    f=figure()
    s2_color=(.3,0,.7)
    s3_color=(1,0,0)
    
    ##Plot signals##
    line_kwargs=dict(lw=2,ls='-')
    abs(s2_sic_dp*ff_sic).plot(color=s2_color,**line_kwargs)
    abs(s3_sic_dp*ff_sic).plot(color=s3_color,**line_kwargs)
    abs(s2_sio2_dp*ff_sio2).plot(color=s2_color,**line_kwargs)
    abs(s3_sio2_dp*ff_sio2).plot(color=s3_color,**line_kwargs)
    ylim(0,5)
    
    ##Plot phases##
    twinx()
    line_kwargs=dict(lw=2,ls='--')
    ((s2_sic_dp*ff_sic).phase/(2*pi)).plot(color=s2_color,**line_kwargs)
    ((s3_sic_dp*ff_sic).phase/(2*pi)).plot(color=s3_color,**line_kwargs)
    ((s2_sio2_dp*ff_sio2).phase/(2*pi)).plot(color=s2_color,**line_kwargs)
    ((s3_sio2_dp*ff_sio2).phase/(2*pi)).plot(color=s3_color,**line_kwargs)
    ylim(-1,1.5)
    
    ##Now add legend##
    lines=f.axes[0].lines[:2]+f.axes[1].lines[:2]
    labels=['$S_2$','$S_3$','$\Phi_2$','$\Phi_3$']
    sca(f.axes[1])
    gca().set_yticks((-.5,0,.5,1,1.5))
    yticks(fontsize=16)
    ylabel('$\Phi-\Phi_{Au}\,[2\pi]$',fontsize=25,rotation=270)
    l=legend(lines,labels,shadow=True,fancybox=True,loc='upper left')
    for t in l.texts: t.set_fontsize(16)
    
    sca(f.axes[0])
    gca().set_yticks((1,2,3,4,5))
    xticks(fontsize=16)
    yticks(fontsize=16)
    xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    ylabel('$|S|/|S_{Au}|$',fontsize=25)
    axhline(0,ls='--',color='k',lw=1.5)
    
    return f
    
def PlotSiO2NanosphereSpectroscopy(harmonic=3,xlims=[750,1250]):
    
    d={}
    data_path='/Users/alexmcleod/tools/python/NearFieldOptics/Data/SiO2_Nanospheres_Comparison'
    d[50]=cPickle.load(open(os.path.join(data_path,\
                                         'SiO2_Nanosphere_50nm_smoothed_curves.pickle')))
    d[80]=cPickle.load(open(os.path.join(data_path,\
                                         'SiO2_Nanosphere_80nm_smoothed_curves.pickle')))
    d[150]=cPickle.load(open(os.path.join(data_path,\
                                         'SiO2_Nanosphere_150nm_smoothed_curves.pickle')))
    
    data_path='/Users/alexmcleod/tools/python/NearFieldOptics/Data/SiO2_Nanospheres_on_Gold'
    d[100]=cPickle.load(open(os.path.join(data_path,\
                                          'SiO2_NS_Specimen1_smoothed_curves.pickle')))
    
    f=figure()
    axes=grid_axes([1,1],spacing=0,\
                   xstart=.15,xstop=.7,\
                   bottom=.1,top=.95)
    axes.reverse()
    sizes=d.keys(); sizes.sort()
    colors=zip(linspace(1,0,len(sizes)),\
               [0]*len(sizes),\
               linspace(0,1,len(sizes)))
    
    lines=[]; labels=[]
    i=0
    for size,color in zip(sizes,colors):
        sca(f.axes[0])
        (d[size]['raw_amplitude']['S_%i'%harmonic]+.2*i-.1).plot(marker='o',ls='',color=color)
        (d[size]['smoothed_amplitude']['S_%i'%harmonic]+.2*i-.1).plot(lw=3,color=color)
        lines.append(gca().lines[-1]); labels.append('%i nm'%size)
        
        if size!=100:
            sca(f.axes[1])
            raw_phase=d[size]['raw_phase']['S_%i'%harmonic]
            smoothed_phase=d[size]['smoothed_phase']['S_%i'%harmonic]
            #npis=round(smoothed_phase.cslice[xlims[0]]/(2*pi)); dphi=npis*2*pi
            dphi=smoothed_phase.cslice[xlims[0]]
            ((smoothed_phase-dphi)/(2*pi)).plot(lw=3,color=color,ls='--')
            #((raw_phase-npis*2*pi)/(2*pi)).plot(color=color,marker='o',ls='')
        
        i+=1
        
    sca(f.axes[1]); xlabel(''); xlim(*xlims); ylim(-.17,.3)
    yticks([-.1,0,.1,.2,.3], fontsize=16)
    xticks([800,900,1000,1100,1200],[],fontsize=16)
    ylabel('$\Phi_%i-\Phi_{%i\,Au}\,[2\pi]$'%(harmonic,harmonic),fontsize=30)
    l=legend(lines,labels,loc='upper left',fancybox=True,shadow=True)
    for t,color in zip(l.texts,colors): t.set_fontsize(20); t.set_color(color)
    l=axvline(1143,color='k',lw=2,ls='--'); l.set_zorder(0)
    grid()
    sca(f.axes[0]); xlim(*xlims); ylim(0,.65); yticks((arange(7))*.1)
    yticks(fontsize=16)
    xticks([800,900,1000,1100,1200],fontsize=16)
    ylabel('$|S_%i|/|S_{%i\,Au}|$'%(harmonic,harmonic),fontsize=30)
    xlabel('$\omega\,[cm^{-1}]$',fontsize=30)
    grid()
        
def PlotSiCNanoparticleSpectroscopy(harmonic=3,xlims=[720,970],\
                                    amplitude_smoothing=4,phase_smoothing=8):
    
    d={}
    data_path='/Users/alexmcleod/tools/python/NearFieldOptics/Data/SiC_Nanoparticles_Comparison'
    d[1]=cPickle.load(open(os.path.join(data_path,\
                                        '400nmSiC_Main_smoothed_curves.pickle')))
    d[2]=cPickle.load(open(os.path.join(data_path,\
                                         '400nmSiC_SmallChunk2_smoothed_curves.pickle')))
    d[3]=cPickle.load(open(os.path.join(data_path,\
                                        '400nmSiC_SmallChunk_smoothed_curves.pickle')))
    
    f=figure()
    axes=grid_axes([1,1],spacing=0,\
                   xstart=.15,xstop=.7,\
                   bottom=.1,top=.95)
    axes.reverse()
    nums=d.keys()
    colors=zip(linspace(0,1,len(nums)),\
               [0]*len(nums),\
               linspace(1,0,len(nums)))
    
    lines=[]; labels=[]
    for num,color in zip(nums,colors):
        sca(f.axes[0])
        raw_amplitude=d[num]['raw_amplitude']['S_%i'%harmonic]
        raw_amplitude.plot(marker='o',ls='',color=color)
        smoothed_amplitude=smooth(raw_amplitude,window_len=amplitude_smoothing)
        smoothed_amplitude.plot(lw=2,color=color)
        lines.append(gca().lines[-1]); labels.append('%i'%num)
        
        sca(f.axes[1])
        raw_phase=d[num]['raw_phase']['S_%i'%harmonic]
        smoothed_phase=smooth(raw_phase,window_len=phase_smoothing)
        #npis=round(smoothed_phase.cslice[xlims[0]]/(2*pi)); dphi=npis*2*pi
        dphi=smoothed_phase.cslice[xlims[0]]
        ((smoothed_phase-dphi)/(2*pi)).plot(lw=3,color=color,ls='--')
        #((raw_phase-npis*2*pi)/(2*pi)).plot(color=color,marker='o',ls='')
        
    sca(f.axes[1]); xlabel('')
    yticks([0,.05,.1,.15,.2],fontsize=16); ylim(-.03,.23)
    xticks([750,800,850,900,950],[],fontsize=16); xlim(*xlims)
    ylabel('$\Phi_%i-\Phi_{%i\,Au}\,[2\pi]$'%(harmonic,harmonic),fontsize=30)
    l=legend(lines,labels,loc='upper left',fancybox=True,shadow=True)
    for t in l.texts: t.set_fontsize(16)
    l=axvline(1143,color='k',lw=2,ls='--'); l.set_zorder(0)
    grid()
    sca(f.axes[0]); xlim(*xlims); ylim(0,1); yticks([0,.2,.4,.6,.8])
    yticks(fontsize=16)
    xticks([750,800,850,900,950],fontsize=16)
    ylabel('$|S_%i|/|S_{%i\,Au}|$'%(harmonic,harmonic),fontsize=30)
    xlabel('$\omega\,[cm^{-1}]$',fontsize=30)
    grid()
    
    d={}
    data_path='/Users/alexmcleod/tools/python/NearFieldOptics/Data/SiC_Nanoparticles_Comparison'
    d[1]=cPickle.load(open(os.path.join(data_path,\
                                        '300nmSiC_LeftShelf_smoothed_curves.pickle')))
    d[2]=cPickle.load(open(os.path.join(data_path,\
                                         '300nmSiC_Main_smoothed_curves.pickle')))
    d[3]=cPickle.load(open(os.path.join(data_path,\
                                         '300nmSiC_SmallTopChunk_smoothed_curves.pickle')))
    
    f=figure()
    axes=grid_axes([1,1],spacing=0,\
                   xstart=.15,xstop=.7,\
                   bottom=.1,top=.95)
    axes.reverse()
    nums=d.keys()
    colors=zip(linspace(0,1,len(nums)),\
               [0]*len(nums),\
               linspace(1,0,len(nums)))
    
    lines=[]; labels=[]
    for num,color in zip(nums,colors):
        sca(f.axes[0])
        raw_amplitude=d[num]['raw_amplitude']['S_%i'%harmonic]
        raw_amplitude.plot(marker='o',ls='',color=color)
        smoothed_amplitude=smooth(raw_amplitude,window_len=amplitude_smoothing)
        smoothed_amplitude.plot(lw=2,color=color)
        lines.append(gca().lines[-1]); labels.append('%i'%num)
        
        sca(f.axes[1])
        raw_phase=d[num]['raw_phase']['S_%i'%harmonic]
        smoothed_phase=smooth(raw_phase,window_len=phase_smoothing)
        #npis=round(smoothed_phase.cslice[xlims[0]]/(2*pi)); dphi=npis*2*pi
        dphi=smoothed_phase.cslice[xlims[0]]
        ((smoothed_phase-dphi)/(2*pi)).plot(lw=3,color=color,ls='--')
        #((raw_phase-npis*2*pi)/(2*pi)).plot(color=color,marker='o',ls='')
        
    sca(f.axes[1]); xlabel('')
    yticks([-.5,0,.05,.1,.15],fontsize=16); ylim(-.04,.15)
    xticks([750,800,850,900,950],[],fontsize=16); xlim(*xlims)
    ylabel('$\Phi_%i-\Phi_{%i\,Au}\,[2\pi]$'%(harmonic,harmonic),fontsize=30)
    l=legend(lines,labels,loc='upper left',fancybox=True,shadow=True)
    for t in l.texts: t.set_fontsize(16)
    l=axvline(1143,color='k',lw=2,ls='--'); l.set_zorder(0)
    grid()
    sca(f.axes[0]); xlim(*xlims); ylim(0,.9); yticks([0,.2,.4,.6,.8])
    yticks(fontsize=16)
    xticks([750,800,850,900,950],fontsize=16)
    ylabel('$|S_%i|/|S_{%i\,Au}|$'%(harmonic,harmonic),fontsize=30)
    xlabel('$\omega\,[cm^{-1}]$',fontsize=30)
    grid()


def PlotSicNanoparticleSizeShifts():
    
    amin=10; amax=35; aavg=20.
    obs_sizes=[400,200,100,250,150,80]
    redshifts=[0,-13,-102,7,-30,-121]
    actual_sizes=[size-aavg*2 for size in obs_sizes]
    up_errs=[2*(amax-aavg)]*len(actual_sizes)
    down_errs=[2*(aavg-amin)]*len(actual_sizes)
    
    figure()
    redshifts,actual_sizes=sort_by(redshifts,actual_sizes)
    colors=zip(linspace(1,0,len(redshifts))**.65,\
               [0]*len(redshifts),\
               linspace(0,1,len(redshifts))**.65)
    for color,redshift,actual_size in zip(colors,redshifts,actual_sizes):
        plot([actual_size],[redshift],marker='o',color=color,ls='',markersize=10)
    errorbar(actual_sizes,redshifts,ls='',xerr=[up_errs,down_errs],color='k',lw=2.5,\
             capsize=10)
    axhline(0,ls='--',color='k',lw=1.5)
    
    xlabel('Size $[nm]$',fontsize=25)
    xticks([0,50,100,200,300,400],fontsize=20)
    
    grid()
    
    ylabel('$\Delta \omega\, [cm^{-1}]$',fontsize=28)
    yticks([0,-25,-50,-75,-100,-125],fontsize=20)