from matplotlib.pyplot import *
from numpy import *
from NearFieldOptics import Materials as mat
from common import plotting

def PlotLinscaleRp(Ef=2400,BSTS_thickness=35e-7,a=30e-7,\
                 flims=[250,1450],Nfreqs=500,\
                 qlims=[1e-3,1e2],Nqs=500,surface=True):
    
    global BSTS
    freqs=linspace(flims[0],flims[1],Nfreqs)
    qxs=linspace(qlims[0],qlims[1],Nqs)
    qs=qxs/a
    
    mat.BSTS_Surface_top.chemical_potential=Ef
    mat.BSTS_Surface_bottom.chemical_potential=Ef
    if surface: BSTS=mat.LayeredMedia(mat.BSTS_Surface_top,\
                      (mat.BSTS_35nm_Bulk,BSTS_thickness),\
                      mat.BSTS_Surface_bottom,exit=mat.SiO2_300nm)
    else: BSTS=mat.LayeredMedia((mat.BSTS_35nm_Bulk,BSTS_thickness),exit=mat.SiO2_300nm)
    
    rp=BSTS.reflection_p(freqs,qs);rp.set_axes([freqs,qxs])
    figure();abs(rp).transpose().plot(plotter=imshow,cmap=cm.gnuplot2,colorbar=False)
    clim(.75,1.5)
    
    ylabel('$\omega\,[cm^{-1}]$',fontsize=28); yticks(fontsize=20)
    xlabel('$q\,[a^{-1}]$',fontsize=28); xticks(fontsize=20)
    
    axvline(1,ls='--',color='w',lw=2)
    
    tight_layout()
    
    cbar=colorbar()
    cbar.set_ticks((.75,1,1.25,1.5))
    sca(gcf().axes[1]); ylabel('$\mathrm{abs}[\,r_p\,]$',fontsize=28,rotation=270)

def PlotLogscaleRp(Ef=2400,BSTS_thickness=35e-7,a=30e-7,\
                 flims=[250,1450],Nfreqs=500,\
                 qlims=[1e-3,1e2],Nqs=500,surface=True):
    
    global BSTS
    freqs=linspace(flims[0],flims[1],Nfreqs)
    qxs=logspace(log(qlims[0])/log(10.),log(qlims[1])/log(10.),Nqs)
    qs=qxs/a
    
    mat.BSTS_Surface_top.chemical_potential=Ef
    mat.BSTS_Surface_bottom.chemical_potential=Ef
    if surface: BSTS=mat.LayeredMedia(#(mat.BN_GPR,10e-7),
                                      mat.BSTS_Surface_top,\
                      (mat.BSTS_35nm_Bulk,BSTS_thickness),\
                      mat.BSTS_Surface_bottom,exit=mat.SiO2_300nm)
    else: BSTS=mat.LayeredMedia((mat.BSTS_35nm_Bulk,BSTS_thickness),exit=mat.SiO2_300nm)
    
    rp=BSTS.reflection_p(freqs,qs);rp.set_axes([freqs,qxs])
    figure();abs(rp).transpose().plot(plotter=imshow,cmap=cm.gnuplot2,colorbar=False)
    clim(.75,1.5)
    
    ylabel('$\omega\,[cm^{-1}]$',fontsize=28); yticks(fontsize=20)
    xlabel('$q\,[a^{-1}]$',fontsize=28); xticks(fontsize=20)
    gca().set_xscale('log')
    
    axvline(1,ls='--',color='w',lw=2)
    
    tight_layout()
    
    cbar=colorbar()
    cbar.set_ticks((.75,1,1.25,1.5))
    sca(gcf().axes[1]); ylabel('$\mathrm{abs}[\,r_p\,]$',fontsize=28,rotation=270)
    