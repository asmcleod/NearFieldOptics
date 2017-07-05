import os
import numpy
import cPickle
from scipy import linalg
from common import misc
from common import numerics
from common.baseclasses import ArrayWithAxes as AWA
from common.log import Logger
from numpy import *
from matplotlib.pyplot import *
from scipy.interpolate import interp1d

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

def get_charge_dist(z0=0,N=1000,\
                        L=1,\
                        wavelength=.1,\
                        V_ext='wave',\
                        total_charge=0,\
                        epsilon=-numpy.inf,\
                        smoothing=1,quadrature='linear',\
                        **kwargs):
        
    #Determine quadrature
    #Use explicit quadrature
    if hasattr(quadrature,'__len__') and not \
       isinstance(quadrature,str) and \
       len(quadrature)==2:
        zs,weights=quadrature
        #Make sure xs go from 0 to L
        N=len(zs)
        zs-=zs.min()
        delta_z=zs.max()
        rescaling=L/float(delta_z)
        zs*=rescaling; zs*=rescaling
    #Compute desired quadrature
    elif hasattr(quadrature,'__call__'):
        zs,weights=quadrature(N,span=L,**kwargs)
    else:
        zs,weights=get_quadrature(N,span=L,kind=quadrature,**kwargs)
    
    #Axes
    N=len(zs)
    zs1=numpy.resize(zs,(N,1))
    zs2=numpy.resize(zs,(1,N))
    z_axis=zs+z0
    
    #Interaction kernel
    Self=-log(abs((zs1-zs2)/float(L)))
    for i in range(Self.shape[0]): Self[i,i]=0
    #Ref=image_interaction(zs1,zs2,R1=R1,R2=R2,z0=z0,r=r)

    #Weights#
    weights_row=weights.view()
    weights_row.resize((1,len(weights))) #Give weights row shape
    A=numpy.matrix((Self)*weights_row) #include weights for integral operator
    
    #Damping term (damps second derivative of solution by an amount gamma)
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
    G=(A.T*A+gamma**2*H).getI()*A.T; del H
    
    
    #Inhomogeneous term
    zs=zs1.squeeze()
    if V_ext==None: phi_ext=numpy.zeros(zs.shape)
    elif V_ext=='linear': phi_ext=-zs.copy()/float(R)
    elif V_ext=='point': phi_ext=1/numpy.abs((zs+z0)/float(R))
    elif V_ext=='wave': phi_ext=exp(1j*2*pi*(zs-L/2.)/float(wavelength))*wavelength/(2*pi)
    elif V_ext=='exponential':
        radii=numpy.array(Rs).squeeze()
        phi_ext=numpy.exp(-(z0+zs)/float(exp_decay))\
                *j0(radii/exp_decay)\
                *exp_decay/float(R) #fixed electric field strength at e.g. z=0
    phi_ext=numpy.matrix(phi_ext).T
   
    #internal potential is set by total charge
    weights_col=weights.view()
    weights_col.resize((len(weights),1)) #Give weights column shape
    top=(total_charge+(1-1/float(epsilon))\
                      *numpy.sum(numpy.array(G*phi_ext)*weights_col))
    bottom=numpy.sum(numpy.array(G)*weights_col)
    V0=top/bottom
    phi_tot=V0+phi_ext/float(epsilon)
    
    global g
    g=phi_tot-phi_ext
    
    #Solution
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
    g=AWA(numpy.array(g).squeeze(),\
                                     axes=[z_axis],
                                     axis_names=['Z'])
    
    return f,eps,e