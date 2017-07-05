import os
import numpy
from common import numerics as num
from common.plotting import grid_axes,axes_limits
from common.baseclasses import ArrayWithAxes as AWA
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from NearFieldOptics import Materials as mat

def get_graphene_plasmon_dispersion(freqs,Ef=1500,loss=100,\
                                    material1=mat.Air,\
                                    material2=mat.SiO2,\
                                    enforce_positive=False):
        
    alpha=1/137.04
    if hasattr(freqs,'__len__'): freqs=numpy.array(freqs)
    
    eps1=material1.epsilon(freqs)
    eps2=material2.epsilon(freqs)
    
    qs=2*numpy.pi*(eps1+eps2)*freqs*(freqs+1j*loss)/(4*alpha*Ef)
    if enforce_positive and hasattr(qs,'__len__'):
        qs=qs[qs.real>0]
    
    return qs

def plot_plasmon_characteristics(Efs=[1000,1500,2000,2500],losses=[30,100,200,300],\
                                 substrate=mat.SiO2):
    
    global free_space_wls
    global freqs
    global wls

    freqs=numpy.linspace(400,1600,1000)
    free_space_wls=AWA(1/freqs,axes=[freqs],axis_names=[r'$\omega$']) #in cm
    
    for j,Ef in enumerate(Efs):
        figtitle=r'Graphene Plasmon on $SiO_2$: $E_f=%s\,cm^{-1}$'%Ef
        
        f1=figure(1+2*j); clf()
        t = f1.text(0.5,
                   0.9, figtitle,
                   horizontalalignment='center',
                   fontproperties=FontProperties(size=22))
        axes1=grid_axes(3,spacing=.125/2.,bottom=.2,top=.75)
        
        f2=figure(2+2*j); clf()
        t = f2.text(0.5,
                   0.9, figtitle,
                   horizontalalignment='center',
                   fontproperties=FontProperties(size=22))
        axes2=grid_axes(2,spacing=.125)
        
        for loss in losses:
            
            qs=get_graphene_plasmon_dispersion(freqs,Ef=Ef,loss=loss,\
                                               material2=mat.SiO2)
            qs_pos=(qs.real>0); qs_neg=(qs.real<=0)
            
            qs_real=AWA(qs.real,axes=[freqs],axis_names=[r'$\omega$']) #in 1/cm
            qs_imag=AWA(qs.imag,axes=[freqs],axis_names=[r'$\omega$']) #in 1/cm
            
            wls=AWA(2*numpy.pi/qs_real,axes=[freqs],axis_names=[r'$\omega$']) #in cm
            decays=AWA(2*numpy.pi/qs_imag,axes=[freqs],axis_names=[r'$\omega$'])
            
            figure(1+2*j)
            f1.sca(axes1[0])
            semilogy(free_space_wls[qs_pos]*1e4,\
                 (free_space_wls/wls)[qs_pos],\
                 lw=1.5,label=r'$\Gamma=%s\,cm^{-1}$'%loss) #vs. wls in microns
            wl=wls.cslice[886]
            free_space_wl=free_space_wls.cslice[886]
            axhline(free_space_wl/wl,ls='--',color='k')
            axvline(free_space_wl*1e4,ls='--',color='k')
            if loss==losses[0]:
                text(11.7,5,r'$\lambda_{p,886cm^{-1}}\approx%1.0fnm$'%(wl*1e7),\
                     fontsize=19)
            
            f1.sca(axes1[1])
            semilogy(free_space_wls[qs_pos]*1e4,\
                 (free_space_wls/decays)[qs_pos],\
                 lw=1.5,label=r'$\Gamma=%s\,cm^{-1}$'%loss)
            
            f1.sca(axes1[2])
            plot(free_space_wls[qs_pos]*1e4,\
                 (numpy.real(qs)/numpy.imag(qs))[qs_pos],\
                 lw=1.5,label=r'$\Gamma=%s\,cm^{-1}$'%loss)
            legend(fancybox=True,shadow=True)
            
            figure(2+2*j)
            f2.sca(axes2[0])
            plot(qs_real[qs_pos]/1e3,\
                 freqs[qs_pos],\
                 ls='-',lw=1.5,label=r'$\Gamma=%1.0fcm^{-1}$'%loss)
            
            f2.sca(axes2[1])
            semilogx(qs_imag[qs_pos]/1e3,\
                 freqs[qs_pos],\
                 ls='--',lw=1.5,label=r'$\Gamma=%1.0fcm^{-1}$'%loss)
            
            
        figure(1+2*j)
        ylabels=[r'$\lambda_{air}/\lambda_p$',\
                 r'$\lambda_{air}/\lambda_{decay}$',\
                 r'$Q=\lambda_{decay}/\lambda_p$']
        for i in range(3):
            f1.sca(axes1[i])
            axvline(886,ls='--',color='k')
            xlabel(r'$\lambda_{air}\,\left[\mu m\right]$',fontsize=23)
            ylabel(ylabels[i],fontsize=23)
            #axvspan(Ef, freqs.max(), facecolor='r', alpha=0.25)
            if qs_neg.any():
                axvspan(free_space_wls[qs_neg].min()*1e4,\
                        free_space_wls[qs_neg].max()*1e4,\
                        facecolor='white',zorder=3,alpha=1,hatch='//')
            xlim(free_space_wls.min()*1e4,\
                 free_space_wls.max()*1e4)
            if i!=2: ylim(1,1e3)
            grid()
            
        figure(2+2*j)
        xlabels=[r'$Re\,k\,\left[10^3cm^{-1}\right]$',\
                 r'$Im\,k\,\left[10^3cm^{-1}\right]$']
        for i in range(2):
            f2.sca(axes2[i])
            legend(loc='lower right')
            ylabel(r'$\omega\,\left[cm^{-1}\right]$',fontsize=23)
            xlabel(xlabels[i],fontsize=23)
            axhspan(freqs[qs_neg].min(),\
                    freqs[qs_neg].max(),\
                    facecolor='white',zorder=3,alpha=1,hatch='//')
            grid()