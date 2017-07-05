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
SiO2=mat.SiO2_300nm
mat.SupportedGraphene.get_layers()[1].set_material(SiO2)

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
    B=1+besselcollective(signs,rnms,qnm)#*exp(-qnm_damp*damp_length_nm)
    
    return B

def pizza_rp(freq,q,rnm=1e3,theta=0,theta_cone=theta_cone,gap=1e-8):
    
    entrance=mat.Air
    exit=SiO2
        
    ##Get holder for data, and expanded freq & q##
    freq,q,rp=material_types._prepare_freq_and_q_holder_(freq,q,\
                                                         entrance=entrance)
    qnm=q*1e-7 #q in nm

    screening=0
    eps1=entrance.epsilon(freq)
    eps2=exit.epsilon(freq)*screening+(1-screening)
    
    kz1=entrance.get_kz(freq,q)
    kz2=exit.get_kz(freq,q)

    sigma=mat.SupportedGraphene.get_layers()[0].conductivity(freq)
    mu=mat.SupportedGraphene.get_layers()[0].gamma
    omega=2*pi*freq
    
    alpha=1/137.
    gamma=Graphene.gamma
    Ef=mat.SupportedGraphene.get_layers()[0].chemical_potential
    q_damp=freq/float(Ef)*\
            ((eps2.real+1)*gamma+eps2.imag*freq)/\
             (4*alpha)
    qnm_damp=q_damp*1e-7
    
    B=get_B(rnm,qnm,theta,theta_cone,qnm_damp=qnm_damp)
    
    rp+=(eps2*kz1-eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)*B)/\
        (eps2*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega))

    return mat.ensure_complex(rp)

##This one is based off E_inplane=0 at edge of graphene##
def pizza_rp2(freq,q,rnm=1e3,theta=0,theta_cone=theta_cone,screening=1):
    
    #entrance=mat.Air
    #exit=SiO2
        
    ##Get holder for data, and expanded freq & q##
    #freq,q,rp0=material_types._prepare_freq_and_q_holder_(freq,q,\
    #                                                     entrance=entrance)
    qnm=q*1e-7 #q in nm

    #eps1=entrance.epsilon(freq)
    exit=SiO2
    eps2=exit.epsilon(freq)
    eps2_screened=eps2*screening
    
    #kz1=entrance.get_kz(freq,q)
    #kz2=exit.get_kz(freq,q)

    sigma=mat.SupportedGraphene.get_layers()[0].conductivity(freq)
    Ef=mat.SupportedGraphene.get_layers()[0].chemical_potential
    gamma=mat.SupportedGraphene.get_layers()[0].gamma
    
    alpha=1/137.
    omega=2*pi*freq
    q_damp=2*pi*freq/float(Ef)*\
                ((eps2_screened.real+1)*gamma+eps2_screened.imag*freq)/\
                (4*alpha)
    qnm_damp=q_damp*1e-7
    
    #top=eps2_screened*kz1-eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)
    #bottom=eps2_screened*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(mat.c*omega)
    #rp0+=top/bottom
    
    rp0=mat.SupportedGraphene.reflection_p(freq,q)
    
    B=get_B(rnm,qnm,theta,theta_cone,qnm_damp=qnm_damp)
    #rp_sio2=(eps2*kz1-eps1*kz2)/(eps2*kz1+eps1*kz2)
    rp=rp0*B#+(B-1)*rp_sio2#+2*(1-B)

    return mat.ensure_complex(rp)

##This version does area-averaging##
def pizza_rp3(freq,q,rnm=1e3,theta=0,theta_cone=theta_cone,screening=1,a=30,N=5):
        
    rpG=mat.SupportedGraphene.reflection_p(freq,q,screening=screening)
    rpS=mat.SiO2_Bulk.reflection_p(freq,q)
    eps2_screened=mat.SiO2_Bulk.epsilon(freq)*screening
        
    ##Reshape array parameters##
    if hasattr(rpG,'__len__') and len(rpG):
        rpG=reshape(rpG,rpG.shape+(1,1))
        rpS=reshape(rpS,rpS.shape+(1,1))
        q=reshape(q,q.shape+(1,1))
    
    ##Get graphene damping##
    omega=2*pi*freq
    qnm=q*1e-7 #q in nm
    alpha=1/137.
    gamma=Graphene.gamma
    Ef=Graphene.chemical_potential
    q_damp=2*pi*freq/float(Ef)*\
                 ((eps2_screened.real+1)*gamma+eps2_screened.imag*freq)/\
                 (4*alpha)
    qnm_damp=q_damp*1e-7
    
    a=30;N=30
    ##Get positions in vicinity of tip##
    x,y=rnm*sin(theta*pi/180.),rnm*cos(theta*pi/180.)
    dx,dy=mgrid[-2*a:2*a:1j*N,-2*a:2*a:1j*N]
    dx.resize((1,N,N)); dy.resize((1,N,N))
    
    gauss=exp(-(dx**2+dy**2)/(2*a**2))
    gauss/=sum(gauss)
    rnms=sqrt((x+dx)**2+(y+dy)**2)
    thetas=180/pi*arctan2((x+dx),(y+dy))
    
    global rp_area
    Bs=get_B(rnms,qnm,thetas,theta_cone,qnm_damp=qnm_damp)
    rp_area=(rpG*Bs+(1-Bs))*gauss;
    for i in range(N):
        for j in range(N):
           if thetas[0,i,j]>theta_cone/2.:
               rp_area[:,i,j]=rpS[:,0,0]*gauss[0,i,j]
    
    #xs=(x+dx).squeeze(); print type(xs)
    #print xs.shape
    #ys=(y+dy).squeeze()
    #rp_area=AWA(rp_area,axes=[q,xs[:,0],ys[0,:]],axis_names=['q','X','Y'])

    ##Return weighted sum over area##
    return mat.ensure_complex(sum(sum(rp_area,axis=-1),axis=-1))

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
    result=pizza_rp(freq=1250,q=q,rnm=rnms,theta=thetas,theta_cone=theta_cone)
    #result=get_B(rnm=rnms,qnm=q*1e-7,theta=thetas,theta_cone=theta_cone,qnm_damp=0)
            
    return AWA(result,axes=[xs.squeeze(),ys.squeeze()])

def get_linescan(freq=886,\
                 mu=None,gamma=None,\
                 theta_cone=theta_cone,Nthetas=50,\
                 a=a,zmin=zmin,amplitude=amplitude,\
                 harmonic=3,Nzs=Nzs,\
                 qmin=qmin,qmax=qmax,\
                 Nqs=Nqs,f=None,r_edge=1e3,\
                 graphene_rp=pizza_rp2,\
                 **kwargs): #qmin and qmax are in interpreted in units of zmin
        
    r_func = lambda theta,theta_cone,r_edge : r_edge/cos((theta_cone/2.-theta)/180.*pi)
    d_func = lambda theta,theta_cone,r_edge : r_edge*tan((theta_cone/2.-theta)/180.*pi)
    
    if mu: mat.SupportedGraphene.get_layers()[0].chemical_potential=mu
    if gamma: mat.SupportedGraphene.get_layers()[0].gamma=gamma
    
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

def get_vertscan(freq=886,\
                 mu=None,gamma=None,\
                 theta_cone=theta_cone,depth=1e3,Nrs=50,\
                 a=a,zmin=zmin,amplitude=amplitude,\
                 harmonic=3,Nzs=Nzs,\
                 qmin=qmin,qmax=qmax,\
                 Nqs=Nqs,f=None,\
                 graphene_rp=pizza_rp2,\
                 **kwargs): #qmin and qmax are in interpreted in units of zmin
    
    if mu: mat.SupportedGraphene.get_layers()[0].chemical_potential=mu
    if gamma: mat.SupportedGraphene.get_layers()[0].gamma=gamma
    
    signal_SiO2=tip_model(freq,rp=SiO2.reflection_p,a=a,zmin=zmin,amplitude=amplitude,\
                              harmonic=harmonic,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                              Nzs=Nzs,**kwargs)
    
    signals=[]
    rnms=linspace(0,depth,Nrs)
    theta=0
    for rnm in rnms:
        Logger.write('Working on NF signal for r=%1.2fnm...'%rnm)
        
        pizza_signal=tip_model(freq,rp=graphene_rp,a=a,zmin=zmin,amplitude=amplitude,\
                               harmonic=harmonic,Nqs=Nqs,qmin=qmin,qmax=qmax,\
                               rnm=rnm,theta=theta,theta_cone=theta_cone,Nzs=Nzs,\
                               **kwargs)#,Nthetas_q=Nthetas_q)
        Logger.write('Done getting NF signal.')
        
        signals.append(pizza_signal/signal_SiO2)
    
    result=AWA(signals,axes=[rnms],axis_names=['Distance from Vertex'])
    
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
             smooth_len=30,\
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
        pizza_signal=tip_model(freq,rp=pizza_rp2,a=a,zmin=zmin,amplitude=amplitude,\
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
    
    ##Smooth##
    xlims,ylims=cone_signal.axis_limits
    Nx,Ny=cone_signal.shape
    smooth_x=round(smooth_len*1e-3/(xlims[1]-xlims[0])*Nx)
    smooth_y=round(smooth_len*1e-3/(ylims[1]-ylims[0])*Ny)
    #cone_signal=smooth(cone_signal,axis=0,window_len=smooth_x)
    #cone_signal=smooth(cone_signal,axis=1,window_len=smooth_y)
    
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