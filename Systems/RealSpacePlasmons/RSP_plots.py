import os
from NearFieldOptics.Systems import RealSpacePlasmons as RSP
from common.misc import extract_array
from common.numerical_recipes import smooth
from common.baseclasses import ArrayWithAxes as AWA
from matplotlib.pyplot import *
from numpy import *
from NearFieldOptics import Materials as mat
from NearFieldOptics import TipModels as tip
from common.plotting import grid_axes

font='Trebuchet MS'
rc('font',**{'family':'sans-serif','sans-serif':[font]})

dir=os.path.dirname(__file__)

def Plot95nmDispersion(Efs=[2000,3000,4000],\
                       gamma=100,\
                       keep_ranges=[(0,760),
                                    (840,1030),
                                    (1164,1350)]):
    
    f=figure()
    
    ###Go through fermi levels###
    fs=linspace(650,1350,800)
    qs=linspace(.005,.15,800)
    colors=zip(linspace(.33,1,len(Efs)),\
               [0]*len(Efs),\
               linspace(.66,0,len(Efs)))
    graphene=mat.SupportedGraphene.get_layers()[0]
    for Ef,color in zip(Efs,colors):
        print 'Plotting for Ef=%i...'%Ef
        
        ##Compute rp for supported graphene##
        graphene.gamma=gamma
        graphene.chemical_potential=Ef
        rp=mat.SupportedGraphene.reflection_p(fs,qs*1e7)
        
        ##Plot q,f for subsets##
        print 'Getting dispersion for supported graphene..'
        for i,range in enumerate(keep_ranges):
            rp_subset=rp.cslice[range[0]:range[1]]
            fset=rp_subset.axes[0]
            qset=array([qs[mag==mag.max()] for mag in abs(rp_subset)])
            
            plot(qset,fset,lw=2,color=color)
            if i==0: gca().lines[-1].set_label('$E_f=%i\,cm^{-1}$'%Ef)
            
        ##Compute rp for suspended graphene
        mat.SuspendedGraphene.gamma=gamma
        mat.SuspendedGraphene.chemical_potential=Ef
        rp=mat.SuspendedGraphene.reflection_p(fs,qs*1e7)
        print 'Getting dispersion for suspended graphene..'
        qset=array([qs[mag==mag.max()] for mag in abs(rp)])
        plot(qset,fs,lw=2,color=color,ls='--')
    
    ##Get data##
    data=extract_array(open(os.path.join(dir,'dispersion95nm.txt')))
    qdata,fdata=2*pi/data[:,1],data[:,0]
    qdata=qdata[fdata<=1000]; fdata=fdata[fdata<=1000] #trim bad point
    
    ##Plot data##
    plot(qdata,fdata,ls='',marker='o',color=(0,1,0),\
         markersize=10,markeredgewidth=1.5,label=None)
    errorbar(qdata,fdata,ls='',xerr=[qdata*(1-1/1.2),\
                                     qdata*(1/.8-1)],\
             color='k',lw=1.5)
    
    ##Plot touchups##
    xlim(0,.1); ylim(fs.min(),fs.max())
    legend(loc='upper right',shadow=True,fancybox=True)
    for t in gca().xaxis.get_major_ticks()+gca().yaxis.get_major_ticks():
        t.label1.set_fontsize(16)
        
def Plot300nmDispersion(Efs=[1000,1500,2000],\
                       gamma=100,\
                       keep_ranges=[(0,760),
                                    (840,980),
                                    (1164,1350)]):
    
    f=figure()
    
    ###Go through fermi levels###
    fs=linspace(650,1350,800)
    qs=linspace(.005,.15,800)
    colors=zip(linspace(.33,1,len(Efs)),\
               [0]*len(Efs),\
               linspace(.66,0,len(Efs)))
    graphene=mat.SupportedGraphene.get_layers()[0]
    for Ef,color in zip(Efs,colors):
        print 'Plotting for Ef=%i...'%Ef
        
        ##Compute rp for supported graphene##
        graphene.gamma=gamma
        graphene.chemical_potential=Ef
        rp=mat.SupportedGraphene.reflection_p(fs,qs*1e7)
        
        ##Plot q,f for subsets##
        print 'Getting dispersion for supported graphene..'
        for i,range in enumerate(keep_ranges):
            rp_subset=rp.cslice[range[0]:range[1]]
            fset=rp_subset.axes[0]
            qset=array([qs[mag==mag.max()] for mag in abs(rp_subset)])
            
            plot(qset,fset,lw=2,color=color)
            if i==0: gca().lines[-1].set_label('$E_f=%i\,cm^{-1}$'%Ef)
            
        ##Compute rp for suspended graphene
        mat.SuspendedGraphene.gamma=gamma
        mat.SuspendedGraphene.chemical_potential=Ef
        rp=mat.SuspendedGraphene.reflection_p(fs,qs*1e7)
        print 'Getting dispersion for suspended graphene..'
        qset=array([qs[mag==mag.max()] for mag in abs(rp)])
        plot(qset,fs,lw=2,color=color,ls='--')
    
    ##Get data##
    data=extract_array(open(os.path.join(dir,'dispersion300nm.txt')))
    qdata,fdata=2*pi/data[:,1],data[:,0]
    
    ##Plot data##
    plot(qdata,fdata,ls='',marker='o',color=(0,1,0),\
         markersize=10,markeredgewidth=1.5,label=None)
    errorbar(qdata,fdata,ls='',xerr=[qdata*(1-1/1.2),\
                                     qdata*(1/.8-1)],\
             color='k',lw=1.5)
    
    ##Plot touchups##
    xlim(0,.1); ylim(fs.min(),fs.max())
    legend(loc='upper right',shadow=True,fancybox=True)
    for t in gca().xaxis.get_major_ticks()+gca().yaxis.get_major_ticks():
        t.label1.set_fontsize(16)

default_cmap=cm.jet
#default_cmap=cm.hot

def PlotConeProfiles(mu=1400,gamma=100,freqs=[905,950,1030,1165,1250],\
                     xrange=1e3,yrange=1e3,\
                      Nx=100,Ny=100,theta_cone=60,yscale='linear',\
                      orientation=-1,cmap=default_cmap,**kwargs):

    #RSP.tip_model=RSP.tip.DipoleModel
    RSP.Graphene.chemical_potential=mu
    RSP.Graphene.gamma=gamma
    RSP.mat.SupportedGraphene.get_layers()[0].chemical_potential=mu
    RSP.mat.SupportedGraphene.get_layers()[0].gamma=gamma
    
    f=figure()
    
    global cones
    
    cones=[]
    axes=grid_axes([1]*len(freqs),spacing=0)
    for i,freq in enumerate(freqs):
        print 'Working on frequency %i of %i, f=%1.2f...'%(i+1,len(freqs),freq)
        cone=RSP.get_cone(freq=freq,\
                          theta_cone=theta_cone,\
                              xrange=xrange,yrange=yrange,Nx=Nx,Ny=Ny,\
                              yscale=yscale,\
                              harmonic=3,
                              orientation=orientation,\
                              plot=False,**kwargs)
        cones.append(cone)
        
        sca(f.axes[i])
        abs(cone).plot(cmap=cmap,colorbar=False)
        
        gca().set_aspect('equal')
        if i!=0: xlabel(''); ylabel(''); xticks([])
        clim(1,abs(cone).max())
        ylabel('$\omega=%1.2fcm^{-1}$'%freq)
        title('$E_f=%s,\,\gamma=%s$'%(mu,gamma))
        
    return cones

def PlotMainlandSpectrum(Efs=[800,900,1000,1100],\
                         gamma=10,amplitude=58,a=20,\
                         tip_model=tip.SSEQModel2,\
                         screening=1):
    
    mat.SupportedGraphene.get_layers()[0].gamma=gamma
    mat.SupportedGraphene.get_layers()[1].set_material(mat.SiO2_Bulk)
    
    ##Tip parameters##
    amplitude=58
    qmin=1e-1
    
    #Some sensical parameters#
    if tip_model==tip.SSEQModel2:
        #tip._SSEQModel2_.response=1.4*exp(1j*pi/20.)
        Nqs=72*2*2
        zmin=1
        a=30
        Nzs=30
        qmax=10
    elif tip_model==tip.ExtendedMonopoleModel or tip_model==tip.DipoleModel:
        tip._ExtendedMonopoleModel_.response=1.4*exp(1j*pi/20.)
        a=30. #35
        zmin=a*.7 #17
        Nqs=400
        Nzs=30
        qmax=10
    
    freqs=linspace(850,1250,75)
    colors=zip(linspace(0,1,len(Efs)),\
               [0]*len(Efs),\
               linspace(1,0,len(Efs)))
    
    f=figure()
    blobdata=extract_array(open(os.path.join(dir,'Blobintensity.txt')))
    freqs_data=blobdata[:,0]
    s3_graph_data=blobdata[:,2]
    s3_sio2_data=blobdata[:,3]
    plot(freqs_data,abs(s3_graph_data/s3_sio2_data),marker='o',color='g',ls='',label='$\mathrm{Data}$')
    xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    ylabel('$|S_3|/|S_{3\,SiO_2}$',fontsize=25)
    
    global s3s
    s3s={}
    print 'Computing SiO2 spectrum...'
    s3_sio2=tip_model(freqs,rp=mat.SiO2_Bulk.reflection_p,\
                                amplitude=amplitude,Nzs=Nzs,harmonic=3,\
                                normalize_to=None,zmin=zmin,a=a,Nqs=Nqs)
    for Ef,color in zip(Efs,colors):
        print 'Computing Ef=%s spectrum...'%Ef
        mat.SupportedGraphene.get_layers()[0].chemical_potential=Ef
        s3_graph=tip_model(freqs,rp=mat.SupportedGraphene.reflection_p,\
                                amplitude=amplitude,Nzs=Nzs,harmonic=3,\
                                normalize_to=None,zmin=zmin,a=a,Nqs=Nqs,\
                                screening=1)
        s3_graph/=s3_sio2
        s3s[Ef]=s3_graph
        abs(s3_graph).plot(color=color,lw=2,label='$%scm^{-1}$'%Ef)
    
    l=legend(loc='best',fancybox=True,shadow=True)
    
    return s3s

def PlotBlobMaxima(Nfreqs=50,cone_angle=60,\
                    gamma=10,mu=900,screening=1,f=None,
                    tip_model=tip.DipoleModel,SiO2=mat.SiO2_Bulk,**kwargs):
    
    freqs=linspace(850,1250,Nfreqs)
    
    if not f:
        f=figure()
        blobdata=extract_array(open(os.path.join(dir,'Blobintensity.txt')))
        freqs_data=blobdata[:,0]
        s3_blob_data=blobdata[:,1]
        s3_sio2_data=blobdata[:,3]
        plot(freqs_data,s3_blob_data/s3_sio2_data,marker='o',color=(0,1,0),ls='',label='$\mathrm{Data}$')
        xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
        ylabel('$|S_{3\,\mathrm{max}}|/|S_{3\,SiO_2}|$',fontsize=25)
    #return
    
    ##Tip parameters##
    amplitude=58
    qmin=1e-1
    
    #Some sensical parameters#
    if tip_model==tip.SSEQModel2:
        #tip._SSEQModel2_.response=1.4*exp(1j*pi/20.)
        Nqs=72*2*2
        zmin=1
        a=30
        Nzs=30
        qmax=10
    elif tip_model==tip.ExtendedMonopoleModel or tip_model==tip.DipoleModel:
        tip._ExtendedMonopoleModel_.response=1.4*exp(1j*pi/20.)
        a=20. #35
        zmin=a*.7 #17
        Nqs=400
        Nzs=30
        qmax=10
    
    RSP.mat.SupportedGraphene.get_layers()[1].set_material(SiO2)
    RSP.mat.SupportedGraphene.get_layers()[0].chemical_potential=mu
    RSP.mat.SupportedGraphene.get_layers()[0].gamma=gamma
    RSP.Graphene.gamma=gamma
    RSP.Graphene.chemical_potential=mu
    
    global s3_sio2
    s3_sio2=tip_model(freqs,rp=SiO2.reflection_p,a=a,zmin=zmin,amplitude=amplitude,\
                              harmonic=3,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                              Nzs=Nzs,**kwargs)
    
    global s3_lines
    s3_lines=[]
    theta=0
    rnms=linspace(2,200,50) #Line cut down center of cone from tip
    blob_maxima=[]
    for i,freq in enumerate(freqs):
        
        print 'Progress: %1.2f'%(i/float(len(freqs))*100)
        
        s3_line=array([tip_model(freq,rp=RSP.pizza_rp2,a=a,zmin=zmin,amplitude=amplitude,\
                                 harmonic=3,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                 Nzs=Nzs,rnm=rnm,theta=theta,theta_cone=60,screening=screening,**kwargs) \
                       for rnm in rnms])
        s3_lines.append(s3_line)
        #s3_line=smooth(s3_line,window_len=10)
        blob_maxima.append(abs(s3_line).max()/abs(s3_sio2[i]))
        
    blob_maxima=array(blob_maxima)
    plot(freqs,blob_maxima,lw=2)
    
    return AWA(blob_maxima,axes=[freqs],axis_names=['Frequency'])
    
def PlotBlobMaximaOverMainland(Nfreqs=50,cone_angle=60,\
                               gamma=10,mu=900,screening=1,\
                               tip_model=tip.DipoleModel,f=None,**kwargs):
    
    freqs=linspace(850,1250,Nfreqs)
    
    if not f:
        f=figure()
        blobdata=extract_array(open(os.path.join(dir,'Blobintensity.txt')))
        freqs_data=blobdata[:,0]
        s3_blob_data=blobdata[:,1]
        s3_graph_data=blobdata[:,2]
        plot(freqs_data,s3_blob_data/s3_graph_data,marker='o',color=(0,1,0),ls='',label='$\mathrm{Data}$')
        xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
        ylabel('$|S_{3\,\mathrm{max}}|/|S_{3\,\mathrm{Bulk}}|$',fontsize=25)
    #return
    
    ##Tip parameters##
    amplitude=58
    qmin=1e-1
    
    #Some sensical parameters#
    if tip_model==tip.SSEQModel2:
        #tip._SSEQModel2_.response=1.4*exp(1j*pi/20.)
        Nqs=72*2*2
        zmin=1
        a=30
        Nzs=30
        qmax=10
    elif tip_model==tip.ExtendedMonopoleModel or tip_model==tip.DipoleModel:
        tip._ExtendedMonopoleModel_.response=1.4*exp(1j*pi/20.)
        a=30. #35
        zmin=a*.7 #17
        Nqs=400
        Nzs=30
        qmax=10
    
    mat.SupportedGraphene.get_layers()[1].set_material(mat.SiO2_Bulk)
    mat.SupportedGraphene.get_layers()[0].chemical_potential=mu
    mat.SupportedGraphene.get_layers()[0].gamma=gamma
    RSP.Graphene.gamma=gamma
    RSP.Graphene.chemical_potential=mu
    
    global s3_graph
    s3_graph=tip_model(freqs,rp=mat.SupportedGraphene.reflection_p,a=a,zmin=zmin,amplitude=amplitude,\
                              harmonic=3,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                              Nzs=Nzs,**kwargs)
    
    global s3_lines
    s3_lines=[]
    theta=0
    rnms=linspace(2,200,200) #Line cut down center of cone from tip
    blob_maxima=[]
    for i,freq in enumerate(freqs):
        
        print 'Progress: %1.2f'%(i/float(len(freqs))*100)
        
        s3_line=array([tip_model(freq,rp=RSP.pizza_rp2,a=a,zmin=zmin,amplitude=amplitude,\
                                 harmonic=3,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                 Nzs=Nzs,rnm=rnm,theta=theta,theta_cone=60,screening=screening,**kwargs) \
                       for rnm in rnms])
        s3_lines.append(s3_line)
        #s3_line=smooth(s3_line,window_len=10)
        blob_maxima.append(abs(s3_line).max()/abs(s3_graph[i]))
        
    blob_maxima=array(blob_maxima)
    plot(freqs,blob_maxima,lw=2)
    
    return AWA(blob_maxima,axes=[freqs],axis_names=['Frequency'])
    
    
def PlotQFactor(Efs=[900,1100,1300],gamma=55,dgamma=45,screening=1):
    
    figure()
    for Ef in Efs:
        RSP.Graphene.chemical_potential=Ef
        RSP.Graphene.gamma=gamma
        
        eps1=entrance.epsilon(freq)
        eps2=exit.epsilon(freq)
        sigma=Graphene.conductivity(freq)
        q=1j*(eps2+eps1)/(4*pi*sigma/(mat.c*omega))
        