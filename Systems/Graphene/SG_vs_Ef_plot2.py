import os
import cPickle
from common.numerics import Spectrum
from numpy import *
import numpy as np
from matplotlib.pyplot import * 
import matplotlib.lines as lines
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from NearFieldOptics import Materials as mat
from NearFieldOptics import TipModels as tip
from mpl_toolkits.axes_grid.inset_locator import inset_axes

data_dir='%s/tools/python/NearFieldOptics/Data/GOSandSiO2Layers'%os.environ.get('HOME')
os.chdir(data_dir)
SGoverGOS=cPickle.load(open('Fei_SGOverGOS.pickle'))
GOSspectrum=cPickle.load(open('sio2gosexpdata.pickle'))['300 + GOS']
SGoverSi=cPickle.load(open('Fei_SGOverSi.pickle'))

expSGoverGOS_P3=cPickle.load(open('Fei_SGOverSi_P3.pickle'))['P31']
expGOSoverSi_P3=Spectrum(cPickle.load(open('sio2gosexpdata.pickle'))['300 + GOS']['P3'])
limiting_freqs=expSGoverGOS_P3.axes[0]
expGOSoverSi_P3=expGOSoverSi_P3.interpolate_axis(limiting_freqs,axis=0)
expSGoverSi_P3=expSGoverGOS_P3-expGOSoverSi_P3*pi
expSGoverSi_P3=expSGoverSi_P3-expSGoverSi_P3.min()

mat.SuspendedGraphene.medium=mat.Air
mat.SuspendedGraphene.medium2=mat.Air
mat.SuspendedGraphene.gamma=50


##To run it, do:
#compute()
#plot()

bestsegs={}
othersegs = []

def compute():
    
    global bestsegs
    global othersegs
    
    models={#'DP':tip.DipoleModel,\
            'EMM':tip.ExtendedMonopoleModel}#,\
            #'SSEQ':tip.SSEQModel}
    a={'DP':25,\
       'EMM':25,\
       'SSEQ':25}
    zmins={'DP':.7*a['DP'],\
           'EMM':.8*a['EMM'],\
           'SSEQ':2}
    amplitudes={'DP':30,\
                'EMM':60,\
                'SSEQ':50}
    Nzs={'DP':50,\
         'EMM':50,\
         'SSEQ':50} #Fewer zs necessary to accurately demodulate SSEQ, since nicer approach curve (it's also slower)
    Nqs={'DP':500,\
         'EMM':500,\
         'SSEQ':72}
    bestlines = {}#LineCollection(segments, cmap=cmap, norm=norm)
    otherlines= {}
    SG_chemical_potentials=[595,645,695]
    bestmu = 645
    bestgamma = 30
    SG_gammas=[10,30,50]
    freqs=linspace(800,1500,75)
    s3s_GOS={}
    lines1 = {}
    #bestsegs = np.zeros((len(SG_chemical_potentials), freqs.size, 2), float)
    
    for model_name,model in models.iteritems():
        for SG_chemical_potential in SG_chemical_potentials:
            muidx = SG_chemical_potentials.index(SG_chemical_potential)
            for SG_gamma in SG_gammas:
                    gammaidx = SG_gammas.index(SG_gamma) 
                    mat.SuspendedGraphene.gamma=SG_gamma
                    mat.SuspendedGraphene.chemical_potential=SG_chemical_potential
                    s3_SG=model(freqs,rp=mat.SuspendedGraphene.reflection_p,\
                                zmin=zmins[model_name],\
                                a=a[model_name],\
                                Nqs=Nqs[model_name],\
                                amplitude=amplitudes[model_name],\
                                harmonic=3,\
                                normalize_to=mat.Si.reflection_p)
                    # if model_name == 'SSEQ':     s3_SG=s3_SG*3.5
                    #lines1[muidx,gammaidx] = abs(s3_SG)
                    #print lines[0,0]
                    if SG_gamma == bestgamma:
                        bestsegs[SG_chemical_potential]=s3_SG
                        #bestsegs[muidx,:,1] = abs(s3_SG[:])
                        #bestsegs[muidx,:,0] = freqs[:]
                    else:
                        othersegs.append(s3_SG)
                        #othersegs[gammaidx,muidx,:,1] = abs(s3_SG[:])
                        #othersegs[gammaidx,muidx,:,0] = freqs[:]
                #else:
                #===================================================================
                # mat.SuspendedGraphene.gamma=SG_gammas[0]
                # mat.SuspendedGraphene.chemical_potential=SG_chemical_potential
                # s3_SG=model(freqs,rp=mat.SuspendedGraphene.reflection_p,\
                #                zmin=zmins[model_name],\
                #                a=a[model_name],\
                #                Nqs=Nqs[model_name],\
                #                amplitude=amplitudes[model_name],\
                #                harmonic=3,\
                #                normalize_to=mat.Si.reflection_p)
                # if model_name == 'SSEQ': s3_SG=s3_SG*3.5
                # abs(s3_SG).plot(wavelength=False,\
                #                                        lw=1.5,\
                #                                    label=r'$\mu_{SG}=%i cm^{-1}$ %s'%(SG_chemical_potential,model_name))
                #===================================================================
                
                
                
def plot():
    
    global bestsegs
    global othersegs
    
    #bestlc = LineCollection(bestlines,linewidths=1.5)
    #plot(bestlc,wavelength=0)
    #legend(bestlc,loc='best')
    #file1 = open('SG_results.txt','w')
    fig = figure()
    ax = axes()
    
    colors=['b','g','r'].__iter__()
    for SG_chemical_potential,s3 in bestsegs.iteritems():
        abs(s3).plot(wavelength=False,lw=1.5,label=r'$\mu=%i cm^{-1}$'%SG_chemical_potential,color=colors.next())
    #best_segments = LineCollection(bestsegs,colors = [colorConverter.to_rgba(i) \
    #                                                 for i in ('b','g','r','c','m','y','k')])
    #ax.add_collection(best_segments)
    best_lines1 = ax.lines
    for otherseg in othersegs:
        abs(otherseg).plot(wavelength=False,lw=1,color=(.7,.7,.7))
        #other_segment = LineCollection(otherseg,colors = (.7,.7,.7))
        #ax.add_collection(other_segment)
    ax.plot(SGoverSi.axes[0],SGoverSi,\
                       marker='o',color='k',\
                       ls='',label=r'$Data$')
    
    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_ylim(0, 1)
    show()
    legend(loc=2)
    xlabel(r'$\omega\,[cm^{-1}]$',fontsize=25)
    ylabel(r'$|S_{3\,SG}|/|S_{3\,Si}|$',fontsize=25)
    title('Suspended Graphene vs Si',fontsize=18)
    grid()
    
    #Plot phase
    ax_ins=inset_axes(ax,width=3,height=2.3,loc=1)
    colors=['b','g','r'].__iter__()
    for SG_chemical_potential,s3 in bestsegs.iteritems():
        ((s3.phase*pi/180.)%(2*pi)).plot(wavelength=False,lw=1.5,color=colors.next())
    ax_ins.plot(expSGoverSi_P3.axes[0],expSGoverSi_P3,\
                       marker='o',color='k',\
                       ls='')
    xlabel(r'$\omega\,[cm^{-1}]$',fontsize=23)
    ylabel(r'$|\Phi_{3\,SG}|/|\Phi_{3\,Si}|$',fontsize=23)
    grid()
    
    #    print line1 
    #    np.savetxt('SG_results.txt',line1)