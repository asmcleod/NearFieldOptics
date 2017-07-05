import os
import numpy
import cPickle
from scipy import linalg
from scipy.special import j0,j1
from common import misc
from common import numerics
from common import numerical_recipes as numrec
from common.baseclasses import ArrayWithAxes as AWA
from common.log import Logger
from matplotlib.pyplot import *
from scipy.interpolate import interp1d
from mpmath import fp,ellipk,ellipe
from mpmath.calculus.quadrature import TanhSinh, GaussLegendre

dir=os.path.dirname(__file__)
os.chdir(dir)

ellipk=numpy.vectorize(ellipk,otypes=[numpy.complex])
ellipe=numpy.vectorize(ellipe,otypes=[numpy.complex])

def make_ellip_interpolators(xmin=.1,xmax=100000,N=100000,remake=False):
    
    filename='ellip_database.pickle'
    if remake or not os.path.isfile(filename):
        print 'Building ellip interpolator objects:\nxmin=%s, xmax=%s, N=%s'%(xmin,xmax,N)
        xpos=numpy.logspace(numpy.log(xmin)/numpy.log(10.),\
                            numpy.log(xmax)/numpy.log(10.),\
                            N)
        xneg=-xpos[::-1]
        x=numpy.hstack((xneg,xpos))
        
        ellipk_database=AWA(ellipk(x),axes=[x])
        ellipe_database=AWA(ellipe(x),axes=[x])
        file=open(filename,'w')
        cPickle.dump({'ellipk':ellipk_database,\
                      'ellipe':ellipe_database},file)
        file.close()
    else:
        print 'Retrieving ellip interpolators'
        file=open(filename)
        d=cPickle.load(file); file.close()
        ellipk_database=d['ellipk']
        ellipe_database=d['ellipe']
    
    ellipk_interpolator=interp1d(x=ellipk_database.axes[0],\
                                 y=ellipk_database,\
                                 kind='slinear')
    ellipe_interpolator=interp1d(x=ellipe_database.axes[0],\
                                 y=ellipe_database,\
                                 kind='slinear')
    
    return ellipk_interpolator,ellipe_interpolator

ellipk_interpolator,ellipe_interpolator=make_ellip_interpolators(remake=False) #just load

def interp_ellipk(x):
    
    Logger.write('Using interpolation from database to compute elliptic K function...')
    min=ellipk_interpolator.x.min()
    max=ellipk_interpolator.x.max()
    
    inside=(x>min)*(x<max)
    outside=(x<=min)+(x>=max)
    
    x_temp=x.copy()
    x_temp[outside]=(min+max)/2. #something safely inside
    y_k=ellipk_interpolator(x_temp)
    
    if outside.any():
        x_outside=x[outside]
        print 'Fraction of argument that could not be interpolated: %s'%(len(x_outside)/\
                                                                         float(numpy.prod(x.shape)))
        y_k_outside=ellipk(x_outside)
        y_k[outside]=y_k_outside
        
    print 'Done.'
    
    return y_k

def interp_ellipe(x):
    
    Logger.write('Using interpolation from database to compute elliptic E function...')
    min=ellipe_interpolator.x.min()
    max=ellipe_interpolator.x.max()
    
    inside=(x>min)*(x<max)
    outside=(x<=min)+(x>=max)
    
    x_temp=x.copy()
    x_temp[outside]=(min+max)/2. #something safely inside
    y_e=ellipe_interpolator(x_temp)
    
    if outside.any():
        x_outside=x[outside]
        print 'Fraction of argument that could not be interpolated: %s'%(len(x_outside)/\
                                                                         float(numpy.prod(x.shape)))
        y_e_outside=ellipe(x_outside)
        y_e[outside]=y_e_outside
        
    print 'Done.'
    
    return y_e

def get_quadrature(N,zmin=1e-3,L=10,kind='GL',prec=4,**kwargs):
    """Use kwarg *beta* for decay constant of exponential quadrature, in units of L."""
    
    global zs,weights
    
    if kind in ['GL','TS']:
    
        if kind=='GL': quadrature=GaussLegendre(fp)
        if kind=='TS': quadrature=TanhSinh(fp)
    
        deg=int(numpy.floor(numpy.log(N)/numpy.log(2)))
        nodes=quadrature.calc_nodes(deg,prec)
        nodes=quadrature.transform_nodes(nodes,a=zmin,b=L)
        
        zs,weights=zip(*nodes)
        zs,weights=misc.sort_by(zs,weights)
        zs=numpy.array(zs)
        weights=numpy.array(weights)
    
    if kind=='linear' or kind is None:
        
        zs=numpy.linspace(zmin,L,N)
        weights=numpy.array([L/float(N)]*int(N))
        
    elif kind=='exponential':
        
        exkwargs=misc.extract_kwargs(kwargs,beta=1)
        beta=exkwargs['beta']
        M=N-1
        zs=L/float(beta)*numpy.log(M/(M-numpy.arange(M)*(1-numpy.exp(-beta))))
        zs=numpy.array(list(zs)+[L])
        weights=numpy.diff(zs)
        weights=numpy.array(list(weights)+[weights[-1]])
    
    elif kind=='double_exponential':
        
        exkwargs=misc.extract_kwargs(kwargs,beta=1)
        beta=exkwargs['beta']
        if N%2==0: N+=1
        M=N-1
        pref=2/float(M)*(numpy.exp(beta/2.)-1)
        
        js_lower=numpy.arange(numpy.ceil(M/2.))
        xs_lower=L*(1/2.-1/float(beta)*numpy.log(numpy.exp(beta/2.)-pref*js_lower))
        
        js_upper=numpy.arange(numpy.ceil(M)-numpy.ceil(M/2.))
        xs_upper=L*(1/2.+1/float(beta)*numpy.log(1+pref*js_upper))
        
        zs=numpy.array(list(xs_lower)+list(xs_upper)+[L])
        weights=numpy.diff(zs)
        weights=numpy.array(list(weights)+[weights[-1]])
    
    elif kind=='simpson':
        
        if N%2==0: N+=1
        zs=numpy.linspace(zmin,L,N)
        weights=numpy.zeros((N,))+2
        weights[numpy.arange(N)%2==0]=4
        weights[0]=1; weights[-1]=1
        weights*=L/float(N-1)/3.
        
    return zs,weights

def get_radii(zs,L,taper_angle=20,geometry='cone'):
    
    #Establish location of tip and make tip coordinates
    R=1
    z0=0
    global tip_coords
    tip=(zs>=z0)*(zs<=(z0+L))
    tip_coords=zs-zs[tip].min()
    
    if geometry in ['cone','cylinder']:
        if geometry=='cone':
            shaft_coord=R*(1-numpy.sin(numpy.deg2rad(taper_angle)))
            shaft=tip*(tip_coords>shaft_coord)
            apex=tip*(tip_coords>=0)*(tip_coords<=shaft_coord)
            apex_radius=numpy.sqrt(R**2-(R-tip_coords)**2)
            apex_radius[numpy.isnan(apex_radius)+numpy.isinf(apex_radius)]=0
        else:
            shaft_coord=0
            shaft=tip
        
        alpha=numpy.tan(numpy.deg2rad(taper_angle))
        Rshaft=R*numpy.cos(numpy.deg2rad(taper_angle))
        Rs=numpy.zeros(zs.shape)+\
           shaft*(Rshaft+alpha*(tip_coords-shaft_coord))
        if geometry=='cone': Rs+=apex*apex_radius
      
    elif geometry=='ellipsoid':
        Rs=numpy.zeros(zs.shape)+\
           tip*R*numpy.sqrt(1-(tip_coords-L/2.)**2/(L/2.)**2)
      
    Rs[Rs==0]=1e-5
    zs,diff_Rs=numerics.differentiate(x=zs,y=Rs)
      
    return Rs,diff_Rs

def get_normals(diff_Rs):
    
    global Nr,Nz
    
    sqrt=numpy.sqrt(diff_Rs**2+1)
    Nr=1/sqrt
    Nz=-diff_Rs/sqrt
    
    return Nr,Nz

def get_evanescent_field(zs,Rs,q=1):
    
    Er=j1(q*Rs)*numpy.exp(-q*zs)
    Ez=j0(q*Rs)*numpy.exp(-q*zs)
    
    return Er,Ez

def get_ring_kernel(zs,Rs):
    """Represents the potential influence due to a line charge
    density a distance *delta_z* away, at which the azimuthally
    symmetric charge distribution has a radius *R*."""
    
    Logger.write('Computing ring kernels over %i x %i points...'%((len(zs),)*2))
    
    #Form index enumerations
    diag_inds=numpy.diag_indices(len(zs))
    triud_inds=numpy.triu_indices(len(zs),k=0)
    triu_inds=numpy.triu_indices(len(zs),k=1) #upper triangle
    tril_inds=[triu_inds[1],triu_inds[0]] #lower triangle
    
    global den1,den2
    K=numpy.zeros((len(zs),)*2,dtype=numpy.float)
    
    #position "2" corresponds to test charge (rows)
    #position "1" corresponds to origin of field (columns)
    zs2=zs.reshape((len(zs),1)); zs1=zs.reshape((1,len(zs)))
    Rs2=Rs.reshape((len(zs),1)); Rs1=Rs.reshape((1,len(zs)))
    
    dr2=(Rs1-Rs2)**2
    dz2=(zs1-zs2)**2
    rmod2=(Rs1+Rs2)**2
    
    den1=numpy.sqrt(dz2+dr2)
    dzs=list(numpy.diff(zs)); dzs=numpy.array(dzs+[dzs[-1]])
    dRs=list(numpy.diff(Rs)); dRs=numpy.array(dRs+[dRs[-1]])
    #fill in diagonal with non-vanishing separation,
    #proportional to geometric mean of z-bins and local radial difference
    den1[diag_inds]=numpy.sqrt(dRs**2+dzs**2)
    arg1=-(4*Rs1*Rs2)/den1**2
    
    den2=numpy.sqrt(dz2+rmod2)
    arg2=+(4*Rs1*Rs2)/den2**2
    
    #Get elliptic function values
    ellipk_triud=interp_ellipk(arg1[triud_inds])
    ellipk2_triud=interp_ellipk(arg2[triud_inds])
    
    K[triud_inds]=(ellipk_triud/den1[triud_inds]+\
                   ellipk2_triud/den2[triud_inds])/numpy.pi
    K[tril_inds]=K[triu_inds]
    
    return K

def get_ring_kernels(zs,Rs):
    """Represents the potential influence due to a line charge
    density a distance *delta_z* away, at which the azimuthally
    symmetric charge distribution has a radius *R*."""
    
    Logger.write('Computing ring kernels over %i x %i points...'%((len(zs),)*2))
    
    #Form index enumerations
    diag_inds=numpy.diag_indices(len(zs))
    triud_inds=numpy.triu_indices(len(zs),k=0)
    triu_inds=numpy.triu_indices(len(zs),k=1) #upper triangle
    tril_inds=[triu_inds[1],triu_inds[0]] #lower triangle
    
    global den1,den2
    
    #position "2" corresponds to test charge (rows)
    #position "1" corresponds to origin of field (columns)
    zs2=zs.reshape((len(zs),1)); zs1=zs.reshape((1,len(zs)))
    Rs2=Rs.reshape((len(zs),1)); Rs1=Rs.reshape((1,len(zs)))
    
    dr2=(Rs1-Rs2)**2
    dz2=(zs1-zs2)**2
    Rmod2=(Rs1+Rs2)**2
    
    dzs=numerics.differentiate(zs)
    dRs=numerics.differentiate(Rs)
    den=numpy.sqrt(dz2+dr2)
    den[diag_inds]=numpy.sqrt(dzs**2+dRs**2)
    arg=(4*Rs1*Rs2)/den**2
    
    #Get elliptic function values on upper diagonal
    ellipk_triud=interp_ellipk(arg[triud_inds])
    ellipe_triud=interp_ellipe(arg[triud_inds])
    
    #All elliptic function results are symmetric
    global ellipkm,ellipkp
    ellipkm,ellipem=[numpy.zeros((len(zs),)*2,dtype=numpy.float) for i in range(2)]
    ellipkm[triud_inds]=ellipk_triud; ellipkm[tril_inds]=ellipkm[triu_inds]
    ellipem[triud_inds]=ellipe_triud; ellipem[tril_inds]=ellipem[triu_inds]
    
    global factor
    factor=1/(dz2+Rmod2-6*Rs1*Rs2)
    Kr=((dz2+Rs1**2-Rs2**2)*ellipem \
        -(dz2+Rmod2-6*Rs1*Rs2)*ellipkm)
    Kr/=2*numpy.pi*(Rs2*den1*(dz2+Rmod2-6*Rs1*Rs2))
    
    Kz=-numpy.sqrt(dz2)*ellipem
    Kz/=numpy.pi*(den1*(dz2+Rmod2-6*Rs1*Rs2))
    
    #Kr[diag_inds]=0; Kz[diag_inds]=0
    
    return Kr,Kz

reuse_kernel=False
zs=None

def get_charge_dist(N=72*4,zmin=1e-3,L=2,geometry='ellipsoid',\
                    taper_angle=20,quadrature='linear',\
                    epsilon=200,q=0,\
                    smoothing=1,**kwargs):
    
    global G,Inv,f,Ez,Er,Kr,Kz
    epsilon=numpy.complex(epsilon)
    
    if not reuse_kernel or zs is None:
        
        #Determine quadrature
        #Use explicit quadrature
        if hasattr(quadrature,'__len__') and not \
           isinstance(quadrature,str) and \
           len(quadrature)==2:
            zs,weights=quadrature
            
        #Compute desired quadrature
        elif hasattr(quadrature,'__call__'):
            Logger.write('Using quadrature "%s".'%quadrature)
            zs,weights=quadrature(N,zmin=zmin,L=L,**kwargs)
            
        else:
            Logger.write('Using quadrature "%s".'%quadrature)
            zs,weights=get_quadrature(N,zmin=zmin,L=L,kind=quadrature,**kwargs)
        
        #Prepare all 1-D arrays
        N=len(zs)
        Rs,diff_Rs=get_radii(zs,L=L,taper_angle=taper_angle,geometry=geometry) #OK
        R_col=Rs.reshape((len(Rs),1))
        
        Nr,Nz=get_normals(diff_Rs) #OK
        Nr_row=Nr.reshape((1,len(Nr))); Nz_row=Nz.reshape((1,len(Nz))) #OK
        
        #Interaction term
        Kr,Kz=get_ring_kernels(zs,Rs)
        W_row=weights.reshape((1,len(weights)))
        
        #pre-multiplying diagonal matrix influences r', post-multiplying influences r
        G=-(epsilon-1)/2.*numpy.matrix(W_row*R_col*(Nr_row*Kr+Nz_row*Kz))
        I=numpy.matrix(numpy.eye(N))
        Inv=numrec.InvertIntegralOperator(I-G,smoothing=smoothing)
        
    else:
        Logger.write('\tReusing earlier-computed interaction kernel in computing charge distribution...')
    
    #Inhomogeneous term
    Logger.write('\tInverting integral equation over N=%i z-values from z=%s to z=%s...'%(N,zs.min(),zs.max()))
    Er,Ez=get_evanescent_field(zs,Rs,q=q)
    #Er=0
    f=numpy.matrix((epsilon-1)/2.*Rs*(Nz*Ez+Nr*Er)).T
    
    #Solution
    charge=Inv*f
    
    #Find some way to implement approximation of error
      
    #Turn to AWA's
    charge=AWA(numpy.array(charge).squeeze(),axes=[zs],axis_names=['Z'])
    
    return charge
