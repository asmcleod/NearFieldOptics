import os
import numpy
import cPickle
from numpy.linalg import solve
from common import numerical_recipes as numrec
from common.log import Logger
from common.baseclasses import ArrayWithAxes as AWA
from matplotlib.pyplot import *
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat

base_dir=os.path.dirname(__file__)

def TestInversion(N=72,z=.1,b=2,x0=.99,taper=10,tip_radius=25,quadrature=numrec.GL,\
                  interpolation='linear',smoothing=1,material=mat.SiO2_300nm2,\
                  freq=1120,geometry='FiniteCone'):
    #GL-quadrature x's using TS-quadrature Lambda values (want high qmax, low qmin) works best
    #With these settings, we get convergence at N=72: b=.75, x0=xs[-1]

    from NearFieldOptics import Materials as mat
    
    load_N=144
    load_quad='TS'
    filename='%sCharge_Taper=%i_Quad=%s_Nzs=288_Nqs=%i.pickle'\
             %(geometry,taper,load_quad,load_N)
    Logger.write('Loading charge data from file "%s"...'%filename)
    try: file=open(os.path.join(tip.charge_data_dir,filename))
    except IOError: Logger.raiseException('No pre-computed charge data was found for parameters:\n'+\
                                          'Nqs=%i\n'%N+'taper=%i'%taper)
    
    #Load data as AWA's with original q's as axes
    charge_data=cPickle.load(file); file.close()
    qs,wqs=charge_data['quadrature']; q1=qs.min(); dq=qs.max()-q1
    Lambda=charge_data['integral_xforms'].transpose() #axes q, s --> s, q
    dipole_moments=charge_data['dipole_moments']
    
    #Get result in qs-space
    Logger.write('Inverting in q-space...')
    Gpref=-qs*numpy.exp(-2*qs*z)
    Qs=numpy.sqrt((2*numpy.pi*freq)**2+(qs/(tip_radius*1e-7))**2)
    #Qs=qs/(tip_radius*1e-7)
    rp=material.reflection_p(freq,Qs)
    GMat=numpy.matrix(numpy.diag(Gpref\
                               *rp))
    LambdaMat=numpy.matrix(Lambda)
    
    alpha=.275
    Lambda0=AWA(qs**(-1-alpha),axes=[qs],axis_names=['q'])
    Lambda0Vec=numpy.matrix(Lambda0).T
    
    I=numpy.matrix(numpy.eye(len(qs)))
    W=numpy.matrix(numpy.diag(wqs))
    kernel=LambdaMat*W*GMat
    result=numpy.array(GMat*solve(I-kernel,Lambda0Vec)).squeeze()
    result=AWA(result,axes=[qs],axis_names=['s'])
    Logger.write('\tDone.')
    
    #Make change of variables
    xs,wxs=numrec.GetQuadrature(xmin=-1,xmax=1,N=N,quadrature=quadrature)
    #x0=xs[-2]
    zstar=1
    a=(2*b+x0)*((1+x0)/(1-x0))**-b\
      /float(2*b*zstar)
      
    global Lambda0Vecp,LambdaMatp,GMatp,rpp,kernelp,qsp,Qsp
    
    q1=qs.min(); deltaq=qs.max()-q1; delta=2*(deltaq/a)**(-1/float(b))
    qsp=a*((1+xs)/(1-xs+delta))**b+q1
    wqsp=a*b*((1+xs)/(2+delta))**b\
            *((2+delta)/(1-xs+delta))**(1+b)\
            /(1+xs)*wxs
    
    Logger.write('Using b=%1.2G, x0=%1.2G, a=%1.2G'%(b,x0,a))
    
    #Develop kernel in xs-space
    Logger.write('\tInverting in x-space...')
    Gprefp=-qsp*numpy.exp(-2*qsp*z)
    Qsp=numpy.sqrt((2*numpy.pi*freq)**2+(qsp/(tip_radius*1e-7))**2)
    #Qsp=qsp/(tip_radius*1e-7)
    rpp=material.reflection_p(freq,Qsp)
    GMatp=numpy.matrix(numpy.diag(Gprefp\
                                  *rpp))
    
    Lambdap=Lambda.interpolate_axis(qsp,axis=0,bounds_error=False,\
                                    extrapolate=True,kind=interpolation)\
                  .interpolate_axis(qsp,axis=-1,bounds_error=False,\
                                    extrapolate=True,kind=interpolation)
    dipole_momentsp=dipole_moments.interpolate_axis(qsp,axis=0)
    LambdaMatp=numpy.matrix(Lambdap)
    
    Lambda0p=AWA(qsp**(-1-alpha),axes=[qsp],axis_names=['q'])
    Lambda0Vecp=numpy.matrix(Lambda0p).T
    
    I=numpy.matrix(numpy.eye(len(xs)))
    Wp=numpy.matrix(numpy.diag(wqsp))
    kernelp=LambdaMatp*Wp*GMatp
    resultp=numpy.array(GMatp*solve(I-kernelp,Lambda0Vecp)).squeeze()
    resultp=AWA(resultp,axes=[qsp],axis_names=['s'])
    
    Logger.write('Done')
    
    d={'qkernel':kernel,\
       'qresult':result,\
       'xkernel':kernelp,\
       'xresult':resultp}
    
    return d
    
def TestInversions(Ns=[72,72*2,72*4,72*8],zs=[10,1,1e-1,1e-2],\
                   b=1.125,x0=.99,taper=10,tip_radius=25,quadrature=numrec.GL,\
                  interpolation='linear',material=mat.SiO2_300nm2,\
                  freq=1120,geometry='FiniteCone',plot=True):
    #x0=.99 and b=.75 are the best for 

    Ns.sort(); Ns.reverse() #max first
    zs.sort(); zs.reverse() #max first
    colors=(['b','c','g','m','y','k','teal','gray','navy']*len(Ns))[:len(Ns)-1]+['r']
    markers=(['+','*','p','s','o']*len(zs))[:len(zs)]
    
    if plot: figure()
    all_ds={}
    for marker,z in zip(markers,zs):
        all_ds_for_z={}
        if plot: axvline(1/float(z),color='k',ls='--')
        
        for color,N in zip(colors,Ns):
            Logger.write('Testing z=%1.2G, N=%i...'%(z,N))
            
            d=TestInversion(N=N,z=z,b=b,x0=x0,taper=taper,tip_radius=tip_radius,quadrature=quadrature,\
                            interpolation=interpolation,material=material,\
                            freq=freq,geometry=geometry)
            
            if plot:
                if N==Ns[-1]: label='z=%1.2G'%z;lw=2;use_marker=marker #This will be N=72, the lowest (last)
                elif N==Ns[0]: label='';lw=2;use_marker=''
                else: label='';lw=2;use_marker=''
                
                if N==Ns[0]:
                    xmax=numpy.abs(d['xresult']).max()
                    qmax=numpy.abs(d['qresult']).max()
                
                (numpy.abs(d['xresult'])/xmax).plot(ls='-',marker=use_marker,color=color,label=label,\
                                                                             plotter=semilogx,lw=lw)
                (numpy.abs(d['qresult'])/qmax).plot(ls=':',marker=use_marker,color=color,label='',\
                                                                             plotter=semilogx,lw=lw)
            
            all_ds_for_z[N]=d
            
        all_ds[z]=all_ds_for_z
        
    leg=legend(title='Coordinate Map:\n'+'$b=%s,\,x_0=%1.2G$'%(b,x0),loc='best',shadow=True,fancybox=True)
        
    return all_ds

def build_quadratures(Nqs_exps=[0,1,2,3,4],\
                     taper_angle=25,geometry='hyperboloid'): #Nqs_exps=4 may be too many
    
    #First build up all charge data for semiinf and finite hyperboloids
    for generator in [tip.build_semiinf_charge_distributions,\
                      tip.build_finite_charge_distributions]:
        
        for quadrature in [tip.numrec.GL,tip.numrec.TS]:
            
            for Nqs_exp in Nqs_exps:
                Logger.write('Using charge generator: %s\n'%generator+\
                             'Quadrature: %s\n'%quadrature+\
                             'Nqs: %i'%Nqs_exp)
                
                Nqs=72*2**Nqs_exp
                generator(Nqs=Nqs,qmin=0,Nzs=72*4,taper_angle=taper_angle,quadrature=quadrature,\
                          geometry=geometry)
                
def test_quadratures(Nqs_exps=[0,1,2,3,4],\
                     freqs=[1000,1100,1125,1150,1200],\
                     taper_angle=25,a=20,\
                     amplitude=80,Nzs=50):
    
    quad_names=['GL','TS']
    model_nos=[1,2,3]
    lss=['-','--',':']
    Nqs=[72*2**Nqs_exp for Nqs_exp in Nqs_exps]
    colors=['b','g','r','c','m','y','k','teal','gray','navy'][:len(Nqs)]
    
    global d,d_geometry,signals_from_quadrature,signals_from_model #for debugging
    d={}
    
    for geometry in ['SemiinfHyperboloid','FiniteHyperboloid']:
        tip.geometry=geometry
        d_geometry={}
        
        for quadrature,quad_name in zip([tip.numrec.GL,tip.numrec.TS],\
                                        quad_names):
            tip.quadrature=quadrature
            signals_from_quadrature=[] #will have following dimensions: model #, Nqs, harmonic #, freq
            
            figure();title('Geometry: %s, Quadrature: %s'%(geometry,quad_name))
            
            for model,ls in zip([tip.LightningRodModel,\
                                          tip.LightningRodModel2,\
                                          tip.LightningRodModel3],lss):
                signals_from_model=[] #will have following dimensions: Nqs, harmonic #, freq
                    
                for Nqs,color in zip(Nqs,colors):
                    Logger.write('Using charge generator: %s\n'%generator+\
                                 'Quadrature: %s\n'%quadrature+\
                                 'Model: %s\n'%model+\
                                 'Nqs: %i'%Nqs)
                    
                    signal_si=tip.LightningRodModel(numpy.mean(freqs),rp=mat.Si.reflection_p,Nqs=Nqs,\
                                                    a=a,zmin=1e-1,amplitude=amplitude,Nzs=Nzs,taper=taper)
                    
                    signals_sio2=tip.LightningRodModel(freqs,rp=mat.SiO2_300nm2.reflection_p,Nqs=Nqs,\
                                                       a=a,zmin=1e-1,amplitude=amplitude,Nzs=Nzs,taper=taper)
                    
                    #Plot the new spectrum#
                    if model is tip.LightningRodModel: label='Nqs=%i'%Nqs
                    else: label=''
                    numpy.abs(signals_sio2[3]/signal_si[3]).plot(marker='+',label=label,ls=ls,color=color)
                    
                    signals_from_Nqs=[signals_sio2[harmonic]/signal_si[harmonic] for harmonic in harmonics]
                    signals_from_model.append(signals_from_Nqs)
                    
                signals_from_quadrature.append(signals_from_model)
            
            signals_from_quadrature=AWA(signals_from_quadrature,\
                                        axes=[model_nos,Nqs,harmonics,freqs],\
                                        axis_names=['Model','Nqs','Harmonic','Frequency'])
        
            d_geometry[quad_name]=signals_from_quadrature
            
        d[geometry]=d_geometry
    
    file=open(os.path.join(base_dir,'QuadratureDiagnostics.pickle'),'w')
    cPickle.dump(d,file); file.close()
    
    return d
        
    