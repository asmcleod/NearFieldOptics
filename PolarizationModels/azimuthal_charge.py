import os
import numpy
import pickle
import time
from scipy import linalg
from scipy.special import j0,j1,ellipk #2020.03.25 ASM: let's try it from scipy!
from common import misc
from common import numerics
from common.baseclasses import ArrayWithAxes as AWA
from common.log import Logger
from matplotlib.pyplot import *
#from mpmath import ellipk
from scipy.interpolate import interp1d,griddata
from scipy.integrate import quad,fixed_quad

basedir=os.path.dirname(__file__)

#ellipk=numpy.vectorize(ellipk,otypes=[numpy.complex])
charge_neutral=True

def make_ellipk_interpolator(xmax=-.0001,xmin=-100000,N=100000,remake=False):

    filename=os.path.join(basedir,'ellipk_database.pickle')
    if remake or not os.path.isfile(filename):
        print('Building ellipk interpolator object:\nxmin=%s, xmax=%s, N=%s'%(xmin,xmax,N))
        x=-numpy.exp(-numpy.linspace(numpy.log(-xmax),
                                     numpy.log(-xmin),
                                     N))
        ellipk_database=AWA(ellipk(x),axes=[x])
        file=open(filename,'w')
        pickle.dump(ellipk_database,file); file.close()
    else:
        print('Retrieving ellipk interpolator')
        file=open(filename)
        ellipk_database=pickle.load(file); file.close()

    ellipk_interpolator=interp1d(x=ellipk_database.axes[0],
                                 y=ellipk_database,
                                 kind='slinear')

    return ellipk_interpolator

#ellipk_interpolator=make_ellipk_interpolator(remake=False) #just load

def interp_ellipk(x):

    print('Using interpolation from database to compute ellipk function...')
    min=ellipk_interpolator.x.min()
    max=ellipk_interpolator.x.max()

    inside=(x>min)*(x<max)
    outside=(x<=min)+(x>=max)

    x_temp=x.copy()
    x_temp[outside]=(min+max)/2. #something safely inside
    y=ellipk_interpolator(x_temp)

    if outside.any():
        x_outside=x[outside]
        print('Fraction of argument that could not be interpolated: %s'%(len(x_outside)/\
                                                                         float(numpy.prod(x.shape))))
        y_outside=ellipk(x_outside)
        y[outside]=y_outside
    print('Done.')

    return y

elliptic_function=interp_ellipk

def get_radii(zs,L=1000,z0=0,R=1,taper_angle=20,geometry='cone',Rtop=0,bow=0):

    #Establish location of tip and make tip coordinates
    global Rs
    Where_Tip=(zs>=z0)*(zs<=(z0+L))
    zs_tip=zs-z0

    Logger.write('Getting geometry for selection "%s"...'%geometry)
    if geometry=='PtSi':

        from NearFieldOptics.PolarizationModels import PtSiTipProfile
        Rs=R*PtSiTipProfile((zs-z0)/float(R),L)

    elif geometry=='sphere':
        L=2*R
        Rs=numpy.zeros(zs.shape)+\
           R*numpy.sqrt(1-(zs_tip-L/2.)**2/(L/2.)**2)

    elif geometry=='ellipsoid':
        b=L/2.
        a=numpy.sqrt(b*numpy.float(R)) #Maintains curvature of 1/R at tip
        Rs=numpy.zeros(zs.shape)+\
           a*numpy.sqrt(1-(zs_tip-b)**2/b**2)

    elif geometry in ['cone','hyperboloid']:

        if geometry=='hyperboloid':
            tan=numpy.tan(numpy.deg2rad(taper_angle))
            a=R/tan**2
            Rs=tan*numpy.sqrt((zs_tip+a)**2-a**2)

        if geometry=='cone':
            ZShft_Bottom=R*(1-numpy.sin(numpy.deg2rad(taper_angle)))
            RShft_Bottom=R*numpy.cos(numpy.deg2rad(taper_angle))

            alpha=numpy.tan(numpy.deg2rad(taper_angle))

            Rs=RShft_Bottom+(zs_tip-ZShft_Bottom)*alpha

            Where_SphBottom=(zs_tip<=ZShft_Bottom)
            Rs[Where_SphBottom]=numpy.sqrt(R**2-(R-zs_tip[Where_SphBottom])**2)
            #Rs[numpy.isnan(Rs)+numpy.isinf(Rs)]=0

        #Add rounded sphere profile to top of cone/hyperboloid
        ZShft_Top=L-Rtop*(1+numpy.sin(numpy.deg2rad(taper_angle)))
        Where_SphTop=(ZShft_Top<zs_tip)*(zs_tip<=L)
        if Where_SphTop.any():

            RShft_Top=Rs[Where_SphTop][0]
            RSph_Top=RShft_Top-Rtop*numpy.cos(numpy.deg2rad(taper_angle))
            ZSph_Top=L-Rtop
            Rs[Where_SphTop]=RSph_Top+numpy.sqrt(Rtop**2-(ZSph_Top-zs_tip[Where_SphTop])**2)

    else: Logger.raiseException('"%s" is an invalid geometry type!'%geometry,exception=ValueError)

    #Make finite only where tip is present
    Rs*=Where_Tip
    Rs[numpy.isnan(Rs)+numpy.isinf(Rs)]=0
    minR=1e-40
    Rs[Rs<=minR]=minR #token small value

    return Rs

def GetCoulKernelQuasiStatic(zs_in,rs_in,zs_out=None,rs_out=None,\
                             ws=None,verbose=True):

    if zs_out is None and rs_out is None: symmetric=True
    else: symmetric=False
    
    if zs_out is None: zs_out=zs_in
    if rs_out is None: rs_out=rs_in
    N_in=len(zs_in)
    N_out=len(zs_out)
    assert len(rs_in)==N_in and len(rs_out)==N_out
    
    Zs_in=numpy.resize(zs_in,(1,N_in))
    if ws is not None: Ws=numpy.reshape(ws,(1,N_in))
    else: Ws=0
    Zs_out=numpy.resize(zs_out,(N_out,1))
    Rs_in=numpy.resize(rs_in,(1,N_in))
    Rs_out=numpy.resize(rs_out,(N_out,1))
    
    NumGrid=-4*Rs_out*Rs_in
    DenGrid=(Zs_out-Zs_in)**2+(Rs_out-Rs_in)**2+Ws/2
    
    if verbose: Logger.write('Evaluating Coulomb Kernel in quasi-static approximation...')
    t1=time.time()
    if symmetric:

        triu_inds=numpy.triu_indices(len(zs))
        tril_inds=[triu_inds[1],triu_inds[0]] #just transpose indices!
        
        #if ws is not None:
        #    numpy.fill_diagonal(DenGrid,(ws/2)**2) #Doesn't matter much as long as not too large/small
    
        #Select only the unique values (above diagonal)
        Den=DenGrid[triu_inds]
        Num=NumGrid[triu_inds]
        Ks=ellipk(Num/Den)
        CoulVals=2/np.pi*Ks/np.sqrt(Den)
        
        CoulKernel=numpy.zeros((N_out,N_in),dtype=numpy.complex)
        CoulKernel[triu_inds]=CoulVals
        CoulKernel[tril_inds]=CoulVals
        
    else:
        KsGrid=ellipk(NumGrid/DenGrid)
        CoulKernel=2/np.pi*KsGrid/np.sqrt(DenGrid)
    
    if verbose: Logger.write('\tTime: %s seconds'%(time.time()-t1))
    CoulKernel=numpy.matrix(CoulKernel)
    
    return CoulKernel

def GetKernels(zs,radii,ws,freq=0,zs2=None,radii2=None):

    #Some default minimum frequency, so default implementation of
    #frequency-dependent kernels doesn't choke on freq=0
    fmin=1e-6
    if freq<=fmin: freq=fmin

    #Get independent vars for kernels
    N=len(zs)
    if zs2 is None: zs2=zs
    zs1=numpy.resize(zs,(N,1))
    zs2=numpy.resize(zs2,(1,N))
    if radii2 is None: radii2=radii
    Rs1=numpy.resize(radii,(N,1))
    Rs2=numpy.resize(radii2,(1,N))

    triu_inds=numpy.triu_indices(len(zs))
    tril_inds=[triu_inds[1],triu_inds[0]] #just transpose indices!

    global XsGrid,YsGrid,Xs,Ys,dRs1,dRs2
    dzs=(zs1-zs2)
    numpy.fill_diagonal(dzs,numpy.diag(dzs)+ws/2.) #Doesn't matter much as long as not too large/small

    XsGrid=freq**2*(dzs**2+Rs1**2+Rs2**2)
    YsGrid=freq**2*(2*Rs1*Rs2)

    #Select only the unique values of Rsq and Bsq (above diagonal)
    Xs=XsGrid[triu_inds]
    Ys=YsGrid[triu_inds]

    #Evaluate Coulomb Kernel#
    Logger.write('Evaluating Coulomb Kernel at freq=%1.2G...'%freq)
    global CoulKernel,FaradKernel
    t1=time.time()
    CoulKernel=numpy.zeros((N,)*2,dtype=numpy.complex)
    CoulVals=numpy.array([calculate_coulomb_kernel_value(X,Y) \
                          for X,Y in zip(Xs,Ys)])*freq
    CoulKernel[triu_inds]=CoulVals
    CoulKernel[tril_inds]=CoulVals
    Logger.write('\tTime: %s seconds'%(time.time()-t1))

    #Evaluate Faraday Kernel
    if freq>fmin:
        Logger.write('Evaluating Faraday Kernel at freq=%1.2G...'%freq)
        global TransFaradKernel,FaradKernel,geom_factor
        dRs=numerics.differentiate(x=zs,y=radii)
        dRs1=numpy.resize(dRs,(N,1))
        dRs2=numpy.resize(dRs,(1,N))
        t1=time.time()
        TransFaradKernel=numpy.zeros((N,)*2,dtype=numpy.complex)
        TransFaradVals=numpy.array([calculate_transverse_faraday_kernel_value(X,Y) \
                                    for X,Y in zip(Xs,Ys)])*freq
        TransFaradKernel[triu_inds]=TransFaradVals
        TransFaradKernel[tril_inds]=TransFaradVals
        TransFaradKernel*=dRs1*dRs2
        geom_factor=1#/numpy.sqrt((1+dRs1**2)*(1+dRs2**2))
        FaradKernel=geom_factor*(CoulKernel\
                                 +TransFaradKernel)
        Logger.write('\tTime: %s seconds'%(time.time()-t1))
    else:
        FaradKernel=numpy.zeros((N,)*2,dtype=numpy.complex)

    CoulKernel=numpy.matrix(CoulKernel)
    FaradKernel=numpy.matrix(FaradKernel)

    return CoulKernel,FaradKernel

def get_quadrature(N,span=1,kind='linear',**kwargs):
    """Use kwarg *beta* for decay constant of exponential quadrature, in units of L."""

    global xs,weights

    if kind=='linear' or kind==None:

        xs=numpy.linspace(0,span,N)
        weights=numpy.array([span/float(N)]*int(N))

    elif kind=='exponential':
        exkwargs=misc.extract_kwargs(kwargs,beta=1)
        beta=exkwargs['beta']
        M=N-1
        xs=span/float(beta)*numpy.log(M/(M-numpy.arange(M)*(1-numpy.exp(-beta))))
        xs=numpy.array(list(xs)+[span])
        weights=numpy.diff(xs)
        weights=numpy.array(list(weights)+[weights[-1]])

    elif kind=='double_exponential':
        exkwargs=misc.extract_kwargs(kwargs,beta=1)
        beta=exkwargs['beta']
        if N%2==0: N+=1
        M=N-1
        pref=2/float(M)*(numpy.exp(beta/2.)-1)

        js_lower=numpy.arange(numpy.ceil(M/2.))
        xs_lower=span*(1/2.-1/float(beta)*numpy.log(numpy.exp(beta/2.)-pref*js_lower))

        js_upper=numpy.arange(numpy.ceil(M)-numpy.ceil(M/2.))
        xs_upper=span*(1/2.+1/float(beta)*numpy.log(1+pref*js_upper))

        xs=numpy.array(list(xs_lower)+list(xs_upper)+[span])
        weights=numpy.diff(xs)
        weights=numpy.array(list(weights)+[weights[-1]])

    elif kind=='simpson':

        if N%2==0: N+=1
        xs=numpy.linspace(0,span,N)
        weights=numpy.zeros((N,))+2
        weights[numpy.arange(N)%2==0]=4
        weights[0]=1; weights[-1]=1
        weights*=span/float(N-1)/3.

    return xs,weights

def get_minimal_smoothing(N): return 9.5e-11*N**1.71

exp_decay=1

reuse_kernel=False

def get_charge_dist(z0=0,N=1000,freq=0,\
                        L=1000,R=1,Rtop=100,taper_angle=0,\
                        geometry='cone',\
                        V_ext='exponential',\
                        total_charge=0,\
                        smoothing=1e-3,quadrature=None,\
                        **kwargs): #typical frequency value has 10 microns v. a=20nm --> f=500 a^-1

    #Some default minimum frequency, so default implementation of
    #frequency-dependent kernels doesn't choke on freq=0
    fmin=1e-10
    if freq<=fmin: freq=fmin

    global radii,zs
    global z_axis
    global weights
    global WUpperTri,A,H,G,RadiativeKernel,phi_ext,reuse_kernel
    
    try: G
    except: reuse_kernel=False

    if not reuse_kernel:

        #Determine quadrature
        #Use explicit quadrature
        if hasattr(quadrature,'__len__') and not \
           isinstance(quadrature,str) and \
           len(quadrature)==2:
            zs,weights=quadrature #Assume the provided values specify length and zmin implicitly
        #Compute desired quadrature
        elif hasattr(quadrature,'__call__'):
            Logger.write('Using quadrature "%s".'%quadrature)
            zs,weights=quadrature(N,span=L,**kwargs)
        else:
            Logger.write('Using quadrature "%s".'%quadrature)
            zs,weights=get_quadrature(N,span=L,kind=quadrature,**kwargs)

        #Axes
        N=len(zs)
        z_axis=zs+z0

        #Interaction kernel
        radii=get_radii(zs,z0=0,L=L,R=R,taper_angle=taper_angle,geometry=geometry,Rtop=Rtop) #z0=0 relative to zs1, which are tip coords
        CoulKernel,FaradKernel=GetKernels(zs,radii,weights,freq=freq)

        #Weights#
        W=numpy.matrix(numpy.diag(weights))

        #Get partial integration matrix
        Logger.write('Calculating charge distribution at freq=%s...'%freq)
        WUpperTri=numpy.matrix([weights]*N).T #weights change along row index
        WUpperTri[numpy.tril_indices(N,-1)]=0 #make lower triangle zero below diagonal
        numpy.fill_diagonal(WUpperTri,.5*weights) #half-weights on diagonal

        #Integrate both variables to obtain radiative kernel
        RadiativeKernel=WUpperTri.T*FaradKernel*WUpperTri

        A=(CoulKernel-(2*numpy.pi*freq)**2*RadiativeKernel)*W #include weights for integral operator

        #Damping term (damps second derivative of solution by an amount gamma)
        Logger.write('Damping second derivative of the solution by an amount %s...'%smoothing)
        diag=6*numpy.eye(N)
        diag[0,0]=diag[-1,-1]=1
        diag[1,1]=diag[-2,-2]=5
        off_diag1=-4*numpy.eye(N)
        off_diag1[0,0]=off_diag1[-1,-1]=-2
        off_diag2=numpy.eye(N)
        H=numpy.matrix(numpy.roll(off_diag2,2,axis=0)+\
                       numpy.roll(off_diag1,1,axis=0)+\
                       diag+\
                       numpy.roll(off_diag1,1,axis=1)+\
                       numpy.roll(off_diag2,2,axis=1))
        H[N-2:N,0:2]=H[0:2,N-2:N]=0 #Don't couple non-adjacent ends

        #Effective inverse of integral operator with smoothing (Green's function)
        gamma=smoothing*get_minimal_smoothing(N)
        G=(A.T*A+gamma**2*H).getI()*A.T

    else:
        Logger.write('Reusing earlier-computed kernel for self energy in computing charge distribution...')

    #Inhomogeneous term - V_ext is produced by a positive charge, producing positive Ez
    if isinstance(V_ext,np.ndarray):
        Logger.write('Using incident potential profile provided by array...')
        phi_ext=np.matrix(np.array(V_ext).squeeze()).T
        
    else:
        Logger.write('Using incident field profile "%s"...'%V_ext)
        if V_ext==None:
            Er,Ez=numpy.zeros(zs.shape),numpy.zeros(zs.shape)
        elif V_ext=='linear':
            Er,Ez=numpy.zeros(zs.shape),numpy.zeros(zs.shape)+1
        elif V_ext=='point':
            Er,Ez=numpy.zeros(zs.shape),1/numpy.abs(z0+zs)**2
        elif V_ext=='exponential':
    
            kappa=R/numpy.float(exp_decay)
            k=2*numpy.pi*freq*R
            Q=numpy.sqrt(k**2+kappa**2)
    
            Ez=j0(Q*radii)*numpy.exp(-kappa*(z0+zs))
            Er=j1(Q*radii)*numpy.exp(-kappa*(z0+zs))*kappa/Q
    
    
        elif 'gaussian' in V_ext:
    
            angle=V_ext[len('gaussian'):]
            if not angle: angle=90
            if angle: angle=float(angle)
            angle*=numpy.pi/180
    
            k=2*numpy.pi*freq*R
            q=k*numpy.sin(angle)
            kz=k*numpy.cos(angle)
    
            Ez=j0(q*radii)
            Er=+1j*numpy.sqrt(k**2-q**2)/q*j1(q*radii)
    
            zpropagation=numpy.exp(-1j*kz*zs)
            Ez=Ez*zpropagation
            Er=Er*zpropagation
    
            WL=1/numpy.float(freq)
            sigma=numpy.sqrt(2)*.42*WL
            envelope=numpy.exp(-(z0+zs)**2/(2*sigma**2))\
                     /(numpy.sqrt(2*numpy.pi)*sigma)
            Ez=Ez*envelope
            Er=Er*envelope
    
        elif 'plane_wave' in V_ext:
    
            angle=V_ext[len('plane_wave'):]
            if not angle: angle=90
            if angle: angle=float(angle)
            angle*=numpy.pi/180
    
            k=2*numpy.pi*freq*R
            q=k*numpy.sin(angle)
            kz=k*numpy.cos(angle)
    
            Ez=j0(q*radii)
            Er=+1j*numpy.sqrt(k**2-q**2)/q*j1(q*radii)#-1j*numpy.sqrt(k**2-q**2)/q*j1(q*radii)
    
            zpropagation=numpy.exp(-1j*kz*zs)
            Ez=Ez*zpropagation
            Er=Er*zpropagation
    
        dRs=numerics.differentiate(y=radii,x=zs)
        Edotted=numpy.matrix(dRs*Er+Ez).T
        phi_ext=-WUpperTri.T*Edotted
        
    Ivec=numpy.matrix([1]*len(phi_ext)).T

    #internal potential is set by total charge
    weights_col=weights.view()
    weights_col.resize((len(weights),1)) #Give weights column shape
    top=numpy.sum(weights_col*numpy.array(G*phi_ext))
    bottom=numpy.sum(weights_col*numpy.array(G*Ivec))
    phi0=top/bottom
    #phi0=0

    global g
    g=phi0-phi_ext

    #Solution
    global f
    f=G*g

    #Error
    global gp
    gp=A*f
    eps=gp-g
    e=numpy.sqrt(numpy.sum(numpy.array(eps)**2))/\
      numpy.sqrt(numpy.sum(numpy.array(g)**2))

    #Turn to AWA's
    f=AWA(numpy.array(f).squeeze(),\
                                     axes=[z_axis],
                                     axis_names=['Z'])
    eps=AWA(numpy.array(eps/g.max()).squeeze(),\
                                     axes=[z_axis],
                                     axis_names=['Z'])

    return f,eps,e

def CoulKernelIm(phi,X,Y):

    Del=numpy.sqrt(X-Y*numpy.cos(phi))

    return numpy.sin(2*numpy.pi*Del)/(2*numpy.pi*Del)

def CoulKernelRe(phi,X,Y):

    Del=numpy.sqrt(X-Y*numpy.cos(phi))

    return numpy.cos(2*numpy.pi*Del)/(2*numpy.pi*Del)

def calculate_coulomb_kernel_value(X,Y,**kwargs):

    ValRe,ErrRe=quad(CoulKernelRe,0,numpy.pi,args=(X,Y),**kwargs)
    ValIm,ErrIm=quad(CoulKernelIm,0,numpy.pi,args=(X,Y),**kwargs)

    return 2*(ValRe+1j*ValIm)

def TransverseFaradKernelIm(phi,X,Y):

    Del=numpy.sqrt(X-Y*numpy.cos(phi))

    return numpy.cos(phi)*numpy.sin(2*numpy.pi*Del)/(2*numpy.pi*Del)

def TransverseFaradKernelRe(phi,X,Y):

    Del=numpy.sqrt(X-Y*numpy.cos(phi))

    return numpy.cos(phi)*numpy.cos(2*numpy.pi*Del)/(2*numpy.pi*Del)

def calculate_transverse_faraday_kernel_value(X,Y,**kwargs):

    ValRe,ErrRe=quad(TransverseFaradKernelRe,0,numpy.pi,args=(X,Y),**kwargs)
    ValIm,ErrIm=quad(TransverseFaradKernelIm,0,numpy.pi,args=(X,Y),**kwargs)

    return 2*(ValRe+1j*ValIm)