from numpy import *
import cPickle
from matplotlib.pyplot import *
from scipy.special import j0
from scipy.interpolate import interp1d,interp2d
from scipy.integrate import simps,trapz
from common.log import Logger
from common.baseclasses import ArrayWithAxes as AWA
from common.numerics import Spectrum
from common.numerical_recipes import smooth
from NearFieldOptics import Materials as mat
from NearFieldOptics.Materials import material_types
from NearFieldOptics import TipModels as tip
tip.verbose=False

##Graphene Parameters##
chemical_potential=900; gamma=5
Graphene=material_types.SingleLayerGraphene(chemical_potential=chemical_potential,\
                                            gamma=gamma)
SiO2=mat.SiO2_Bulk

##Cone parameters##
xsize=1500 #max width of cone in nm
ysize=5000 #max height of cone in nm
theta_cone=60

##Tip parameters##
amplitude=58
qmin=1e-1
qscale='log'
tip_model=tip.DipoleModel
#tip_model=tip.ExtendedMonopoleModel
#tip_model=tip.SSEQModel2

#Some sensical parameters#
if tip_model==tip.SSEQModel2:
    #tip._SSEQModel2_.response=1.4*exp(1j*pi/20.)
    Nqs=72*2
    zmin=1
    a=20
    Nzs=30
    qmax=10
elif tip_model==tip.ExtendedMonopoleModel or tip_model==tip.DipoleModel:
    tip._ExtendedMonopoleModel_.response=1.4*exp(1j*pi/20.)
    a=35. #35
    zmin=17 #17
    Nqs=500
    Nzs=30
    qmax=10

##Bessel parameters##
qnm_max=qmax/float(zmin)
rnm_max=20e3 #maximum tip distance - 20 microns
qr=linspace(0,10*rnm_max*qnm_max,1500000)
#besselmatrix=j0(qr)
#interp_besselmatrix_at=interp1d(qr,besselmatrix)

def get_image_signs_thetas_rs(r=1e3,theta=0,theta_cone=theta_cone,threshold=.1):
    
    image_thetas=[]
    signs=[]
    current_theta=theta_cone; i=1
    while abs(current_theta%360)>=threshold:
        sign=(-1)**i
        #Only add image if it's actually outside main pizza slice
        if abs(current_theta%360)>theta_cone/2.:
            signs.append(sign)
            image_thetas.append(current_theta+sign*theta)
        current_theta+=theta_cone
        i+=1
    
    rs=[abs(r*exp(1j*theta/180.*pi)-r*exp(1j*image_theta/180.*pi)) for image_theta in image_thetas]
    
    return signs,image_thetas,rs

def besselcollective(signs,rnms,qnm):
    
    phase=-1
    phase/=abs(phase)
    
    btot=0
    for sign,rnm in zip(signs,rnms):
        angle=rnm*qnm
        btot+=-phase*sign*j0(rnm*qnm)
        
    return btot

def get_B(rnm,qnm,theta=0,theta_cone=theta_cone,qnm_damp=0):
    
    signs,thetas,rnms=get_image_signs_thetas_rs(rnm,theta,theta_cone)
    
    damp_length_nm=rnm*sin((theta_cone/2.-theta)/180*pi)
    B=1+besselcollective(signs,rnms,qnm)*exp(-qnm_damp*damp_length_nm)
    
    return B

def pizza_rp(freq,q,rnm=1e3,theta=0,theta_cone=theta_cone,screening=1):
    
    entrance=mat.Air
    exit=SiO2
        
    ##Get holder for data, and expanded freq & q##
    freq,q,rp=material_types._prepare_freq_and_q_holder_(freq,q,\
                                                         entrance=entrance)
    qnm=q*1e-7 #q in nm

    eps1=entrance.epsilon(freq)
    eps2=exit.epsilon(freq)*screening+(1-screening)
    
    kz1=entrance.get_kz(freq,q)
    kz2=exit.get_kz(freq,q)

    sigma=Graphene.conductivity(freq)
    omega=2*pi*freq
    
    alpha=1/137.
    gamma=Graphene.gamma
    Ef=Graphene.chemical_potential
    q_damp=freq/float(Ef)*\
            ((eps2.real+1)*gamma+eps2.imag*freq)/\
             (4*alpha)
    qnm_damp=q_damp*1e-7
    
    B=get_B(rnm,qnm,theta,theta_cone,qnm_damp=qnm_damp)
    
    rp+=(eps2*kz1-eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)*B)/\
       (eps2*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega))

    #global bottom
    #bottom=(eps2*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega))**-1
    #bottom=AWA(bottom,axes=[q],axis_names=['q-vector [cm]'])

    return mat.ensure_complex(rp)

graphene_sigma={}
eps_sio2={}

def make_plot(xrange=1e3,yrange=1e3,Nx=50,Ny=50,\
              theta_cone=60,\
              Nthetas_q=100,\
              wavelength=200):
    
    xs=reshape(linspace(0,xrange,Nx),(Nx,1))
    ys=reshape(linspace(0,yrange,Ny),(1,Ny))
    
    rnms=sqrt(xs**2+ys**2)
    thetas=arctan2(ys,xs)*180/pi
    
    q=2*pi/float(wavelength*1e-7)
    
    #result=zeros((Nx,Ny))
    #for i in range(Nx):
    #    for j in range(Ny):
    #        print "progress: %s"%((i*Ny+j)/float(Nx*Ny)*100)
    #        result[i,j]+=reflected_wave_rp(1250,rnm=rnms[i,j],theta=thetas[i,j],theta_cone=theta_cone,q=q,\
    #                                    Nthetas_q=Nthetas_q)
    #result=reflected_wave_rp(1250,rnm=rnms,theta=thetas,theta_cone=theta_cone,q=q,Nthetas_q=Nthetas_q)
    #result=reflected_wave_sum(1250,rnm=rnms,theta=thetas,theta_cone=theta_cone,wavelength=wavelength,theta_q=0)
    result=pizza_rp(freq=1250,q=q,rnm=rnms,theta=thetas,theta_cone=theta_cone,screening=1)
            
    return AWA(result,axes=[xs.squeeze(),ys.squeeze()])

def reflected_wave_sum(freq,rnm=1e3,theta=0,theta_cone=60,wavelength=200,theta_q=0):
    
    theta_rad=pi*theta/180.
    theta_cone_rad=pi*theta_cone/180.
    theta_q_rad=pi*theta_q/180.
    
    #Get damping#
    alpha=1/137.
    gamma=Graphene.gamma
    Ef=Graphene.chemical_potential
    eps2=mat.SiO2.epsilon(freq)
    q_damp=2*pi*((eps2.real+1)*gamma*freq+eps2.imag*freq**2)/\
                (4*alpha*Ef)
    q=2*pi/float(wavelength*1e-7)
    
    default_phase=q*(rnm*1e-7)*cos(theta_rad-theta_q_rad)
    Nimages=int(round(360/float(theta_cone)))
    reflected_wave_sum=0
    for n in range(Nimages):
        image_phase=-q*(rnm*1e-7)*cos(theta_rad-(-1)**n*theta_q_rad-n*theta_cone_rad)
        reflected_wave_sum+=(-1)**n*exp(1j*(default_phase+image_phase))*exp(-.1*q_damp*(rnm*1e-7))
        
    return reflected_wave_sum

def reflected_wave_rp(freq,q,rnm=1e3,theta=0,theta_cone=theta_cone,Nthetas_q=100,smoothing=20):
    
    ##Get holder for data, and expanded freq & q##
    freq,q,rp=material_types._prepare_freq_and_q_holder_(freq,q,\
                                                         entrance=mat.Air)
    
    theta_cone_rad=theta_cone*pi/180.
    theta_rad=theta*pi/180.
    thetas_q_rad=arange(Nthetas_q)/float(Nthetas_q)*2*pi
    
    #Optical properties#
    try:
        sigma=graphene_sigma[freq]
        eps2=eps_sio2[freq]
    except KeyError:
        sigma=Graphene.conductivity(freq)
        eps2=mat.SiO2.epsilon(freq)
        graphene_sigma[freq]=sigma
        eps_sio2[freq]=eps2
    eps1=1
    kz1=mat.Air.get_kz(freq,q)
    kz2=mat.SiO2.get_kz(freq,q)
    omega=2*pi*freq
    
    #Get damping#
    alpha=1/137.
    gamma=Graphene.gamma
    Ef=Graphene.chemical_potential
    q_damp=2*pi*((eps2.real+1)*gamma*freq+eps2.imag*freq**2)/\
                (4*alpha*Ef)
    damping_length_nm=rnm*sin(theta_cone_rad/2.-theta_rad)
    
    #Resize stuff
    if hasattr(rp,'__len__'):
        thetas_q_rad.resize((1,)*rp.ndim+(len(thetas_q_rad),))
        q.resize(q.shape+(1,))
        kz1.resize(kz1.shape+(1,))
        kz2.resize(kz2.shape+(1,))
    
    #Build reflected wave sum as enters rp##
    signs,thetas,rnm_images=get_image_signs_thetas_rs(rnm,theta,theta_cone)
    default_phase=q*(rnm*1e-7)*cos(theta_rad-thetas_q_rad)
    Nimages=int(round(360/float(theta_cone)))
    reflected_wave_sum=0
    for n,rnm_image in zip(range(Nimages),rnm_images):
        image_phase=-q*(rnm*1e-7)*cos(theta_rad-(-1)**n*thetas_q_rad-n*theta_cone_rad)
        reflected_wave_sum+=(-1)**n*exp(1j*(default_phase+image_phase))*exp(-q_damp*(damping_length_nm*1e-7))
        
    #signs,thetas,rnms=get_image_signs_thetas_rs(rnm,theta,theta_cone)
    #default_phase=q*(rnm*1e-7)*cos(theta_rad-thetas_q_rad)
    #Nimages=int(round(360/float(theta_cone)))
    #global reflected_wave_sum
    #reflected_wave_sum=0
    #for n,rnm_image in zip(range(Nimages),rnms):
    #    image_phase=-q*(rnm*1e-7)*cos(theta_rad-(-1)**n*thetas_q_rad-n*theta_cone_rad)
    #    reflected_wave_sum+=(-1)**n*exp(1j*(default_phase+image_phase))#*exp(-2*q_damp*(rnm*1e-7))
        
    thetas_q_axis=thetas_q_rad.squeeze()
    
    rp_integrand=(eps2*kz1-eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)*reflected_wave_sum)/\
                 (eps2*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)*reflected_wave_sum)
    rp+=1/(2*pi)*simps(x=thetas_q_axis,y=rp_integrand,axis=-1)
    #rp=rp_integrand
    
    #rp=smooth(rp,window_len=smoothing,axis=-1)
    
    return mat.ensure_complex(rp)

#Get distances in nm for linescans#
r_perpendicular_func = lambda theta,theta_cone,r_edge : r_edge/cos((theta_cone/2.-theta)/180.*pi)
d_perpendicular_func = lambda theta,theta_cone,r_edge : r_edge*tan((theta_cone/2.-theta)/180.*pi)

def get_linescan(freq=886,\
                 mu=None,gamma=None,\
                 r_func=r_perpendicular_func,\
                 d_func=d_perpendicular_func,\
                 theta_cone=theta_cone,Nthetas=50,\
                 a=a,zmin=zmin,amplitude=amplitude,\
                 harmonic=3,Nzs=Nzs,\
                 qmin=qmin,qmax=qmax,\
                 Nqs=Nqs,f=None,r_edge=1e3,\
                 graphene_rp=pizza_rp,\
                 **kwargs): #qmin and qmax are in interpreted in units of zmin
    
    if gamma: Graphene.gamma=gamma
    if mu: Graphene.chemical_potential=mu
    
    signal_SiO2=tip_model(freq,rp=SiO2.reflection_p,a=a,zmin=zmin,amplitude=amplitude,\
                              harmonic=harmonic,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                              Nzs=Nzs,**kwargs)
    
    signals=[]
    thetas=linspace(theta_cone/2.,0,Nthetas)
    distances=d_func(thetas,theta_cone,r_edge)
    for theta in thetas:
        Logger.write('Working on NF signal for theta=%1.2f...'%theta)
        
        r_tip=r_func(theta,theta_cone,r_edge)
        pizza_signal=tip_model(freq,rp=graphene_rp,a=a,zmin=zmin,amplitude=amplitude,\
                               harmonic=harmonic,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                               rnm=r_tip,theta=theta,theta_cone=theta_cone,Nzs=Nzs,\
                               **kwargs)#,Nthetas_q=Nthetas_q)
        Logger.write('Done getting NF signal.')
        
        signals.append(pizza_signal/signal_SiO2)
    
    result=AWA(signals,axes=[distances],axis_names=['Distance from Edge [nm]'])
    
    if f==None: f=figure()
    else: figure(f.number)
    abs(result).plot()
    
    return result

def get_cone(freq=886,\
             xrange=xsize,yrange=ysize,Nx=200,Ny=400,\
             yscale='linear',\
             theta_cone=theta_cone,\
             a=a,zmin=zmin,amplitude=amplitude,\
             harmonic=3,Nqs=Nqs,qmin=qmin,qmax=qmax,\
             Nthetas_q=100,\
             orientation=1,\
             plot=True,**kwargs):
    
    ##Initialize base arrays##
    if yscale is 'log': ys=logspace(0,log(yrange)/log(10),Ny)
    else: ys=linspace(1,yrange,Ny)
    half_xs=linspace(0,xrange/2.,int(Nx/2.))
    half_signals_arr=ones((len(half_xs),len(ys)), dtype=complex)
    
    ##Decide which grid points are inside cone##
    r_thetas=[]
    index_coords=[]
    for i in range(len(half_xs)):
        xnm=half_xs[i]
        for j in range(len(ys)):
            ynm=ys[j]
            
            theta=arctan2(xnm,ynm)*180/pi
            rnm=sqrt(xnm**2+ynm**2)
            if theta>theta_cone/2.: continue
            r_thetas.append((rnm,theta))
            index_coords.append((i,j))
    
    ##Populate signals at those points##
    signal_sio2=tip_model(freq,rp=SiO2.reflection_p,a=a,zmin=zmin,amplitude=amplitude,\
                      harmonic=harmonic,Nzs=Nzs,Nqs=Nqs,qmin=qmin,qmax=qmax,**kwargs)
    p=1
    for r_theta,index_coord in zip(r_thetas,index_coords):
        rnm,theta=r_theta
        i,j=index_coord
        pizza_signal=tip_model(freq,rp=pizza_rp,a=a,zmin=zmin,amplitude=amplitude,\
                                  harmonic=harmonic,Nzs=Nzs,\
                                  Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                  rnm=rnm,theta=theta,theta_cone=theta_cone,**kwargs)#,Nthetas_q=Nthetas_q)
        half_signals_arr[i,j]=pizza_signal/signal_sio2
        
        progress=p/float(len(index_coords))
        Logger.write('Progress: %1.2f%%'%(progress*100))
        p+=1
    
    all_xs=concatenate((-half_xs[::-1],half_xs))
    cone_signal=concatenate((half_signals_arr[::-1],half_signals_arr),axis=0)
    cone_signal=AWA(cone_signal,axes=[all_xs/1e3,ys/1e3],axis_names=['X','Y'])
    
    if yscale=='log':
        new_ys=linspace(ys.min()/1e3,ys.max()/1e3,10*Ny)
        cone_signal=cone_signal.interpolate_axis(new_ys,axis=1,kind='slinear')
    yaxis=cone_signal.axes[1]
    yaxis=(yaxis.max()-yaxis)
    if orientation==-1: yaxis=yaxis[::-1]
    cone_signal.set_axes(axes=[all_xs/1e3,yaxis])
    
    ##Now we save data##
    filename='graphene_cone_signal.pickle'
    file=open(filename,'w')
    cPickle.dump(cone_signal,file)
    file.close()
    
    ##Now we plot data##
    abs_signal=abs(cone_signal)
    phase=angle(cone_signal); phase=AWA(phase); phase.adopt_axes(cone_signal)
    
    if plot:
        figure(); abs_signal.plot(cmap=cm.hot); xticks((-.5,0,.5))
        xlabel('$x\,[\mu m]$',fontsize=25); ylabel('$y\,[\mu m]$',fontsize=25)
        title('$\mathbf{|S_%i|}$'%harmonic,fontsize=28,verticalalignment='bottom')
        
        figure(); phase.plot(cmap=cm.hot); xticks((-.5,0,.5))
        xlabel('$x\,[\mu m]$',fontsize=25); ylabel('$y\,[\mu m]$',fontsize=25)
        title('$\mathbf{\phi_%i}$'%harmonic,fontsize=28,verticalalignment='bottom')
        
    return cone_signal
