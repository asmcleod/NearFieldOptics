import os
import re
import time
import numpy
import pickle
from common import misc
from common.log import Logger
from numpy.linalg import solve
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline,RectBivariateSpline,interp1d
from common.baseclasses import ArrayWithAxes as AWA
from common.numerics import Spectrum,broadcast_items,number_types,differentiate
from common import numerical_recipes as numrec

from scipy.special import j0,j1,struve
from NearFieldOptics.PolarizationModels import azimuthal_charge as az

basedir=os.path.dirname(__file__)
charge_data_dir=os.path.join(basedir,'ChargeData')

verbose=True

Nqs=244
qmin=1e-3
taper_angle=20
quadrature='TS'
geometry='Hyperboloid'

class TipModel(object):
    """Base workflow for a model. Individual models will want to implement *cls.get_signal*."""
    
    @classmethod
    def __call__(cls,*args,**kwargs):
        
        ##See if we'll want to normalize##
        exkwargs=misc.extract_kwargs(kwargs,normalize_to=None,normalize_at=None)
        normalize_to=exkwargs['normalize_to']
        normalize_at=exkwargs['normalize_at']
        if 'demodulate' in kwargs: demodulate=kwargs['demodulate']
        else: demodulate=True
        
        ##Get signal using whatever model##
        if verbose: Logger.write('Getting signal...')
        signal=cls.get_signal(*args,**kwargs)
        if verbose: Logger.write('Done getting signal')
        
        if normalize_to and demodulate:
            if verbose: Logger.write('Getting normalization signal...')
            kwargs['rp']=normalize_to
            if normalize_at is not None:
                if len(args): args=list(args); args[0]=normalize_at
                else: kwargs['freqs']=normalize_at
            normalization=cls.get_signal(*args,**kwargs)
            
            #Take note that signal will be a dictionary of harmonics
            for key in list(signal.keys()):
                #if signal is a harmonic, normalize
                if 'signal_' in key: signal[key]/=normalization[key]
                else: signal['norm_'+key]=normalization[key]
            
            if verbose: Logger.write('Done normalizing.')
        
        return signal
    
    @classmethod
    def __get_zs__(cls,Nzs,a=20,zs=None,zmin=10,amplitude=50):
        
        if verbose: Logger.write('Building z-values...')
        zmin=zmin/numpy.float(a)
        amplitude=amplitude/numpy.float(a)
        
        if not Nzs: Nzs=50
        cls.ts=numpy.linspace(0,.5,Nzs)
        zs=zmin+amplitude*(1-numpy.cos(2*numpy.pi*cls.ts))
        if verbose: Logger.write('\tDone.')
            
        return zs #Output in units of a
    
    @classmethod
    def __condition_arguments__(cls,freqs,rp,zs,\
                                Nzs=None,a=20,zmin=10,amplitude=50,\
                                Nqs=None,qmin=None,qmax=None,\
                                qscale='linear',\
                                **rpkwargs):
        
        if not hasattr(freqs,'__len__'): freqs=numpy.array([freqs])

        if verbose: Logger.write('Conditioning arguments to model...')

        ##Get zs first##
        if zs is None:
            zs=cls.__get_zs__(Nzs=Nzs,a=a,zmin=zmin,amplitude=amplitude)
        elif verbose: Logger.write('Using provided z-value(s).')
        cls.zs=zs
        
        ##Get qs##
        if not Nqs: Nqs=500
        if not qmin: qmin=1e-2
        if not qmax: qmax=100
        
        if qscale=='linear':
            qs=numpy.linspace(qmin,qmax,Nqs)
        elif qscale=='log':
            qs=numpy.logspace(numpy.log(qmin)/numpy.log(10.),\
                              numpy.log(qmax)/numpy.log(10.),\
                              Nqs)
        
        ##Get rp##
        if hasattr(rp,'__call__'):
            rp=cls.compute_rp(rp,freqs,qs/(a*1e-7),**rpkwargs) #compute rp, with q in units of cm^-1
        else:
            Logger.raiseException('rp must be a number or a function!',\
                                  unless=(isinstance(rp,numpy.ndarray) \
                                          or type(rp) in number_types),\
                                  exception=TypeError)
        
        #Save most recent arguments
        cls.freqs=freqs
        cls.qs=qs
        cls.rp=rp
        
        if verbose: Logger.write('\tDone.')
        
        return rp,freqs,qs,zs
    
    @classmethod
    def __demodulate__(cls,signal,harmonic=3):
        """Takes z-axis as first axis, frequency as final axis."""
        
        if verbose: Logger.write('Demodulating...')
        if verbose: Logger.write('\tSignal shape: %s'%repr(signal.shape))
        
        #Assume z-axis is first axis#
        ts_shape=(len(cls.ts),)+(1,)*(signal.ndim-1)
        ts_grid=cls.ts.reshape(ts_shape)
        signal=simps(numpy.cos(2*numpy.pi*harmonic*ts_grid)*signal,\
                     x=cls.ts,\
                     axis=0)
        
        if verbose: Logger.write('\tDone.')
        
        return signal

    @classmethod
    def compute_rp(cls,rp_function,\
                     freqs=None,qs=None,\
                     **rpkwargs):
        
        if freqs is None:
            if verbose: Logger.write('Using model\'s most recently used frequencies..')
            freqs=cls.freqs
        if qs is None:
            if verbose: Logger.write('Using model\'s most recently used qs..')
            qs=cls.qs
        
        if not rpkwargs: rpkwargs={}
        
        try:
            if verbose: Logger.write('\tTrying to compute rp using provided keyword arguments:\n\t%s'\
                         %rpkwargs)
            rp=rp_function(freqs,qs,**rpkwargs)
        except TypeError:
            if verbose: Logger.write('\tComputing rp without provided keyword arguments.')
            rp=rp_function(freqs,qs)
            
        cls.rp=rp
        
        return rp
        
    @classmethod
    def get_signal(cls,*args,**kwargs): raise NotImplementedError

class _ExtendedMonopoleModel_(TipModel):
    
    response=1.4
    
    @classmethod
    def get_signal(cls,freqs,rp,zs=None,\
                   a=25,zmin=10,amplitude=40,Nzs=None,\
                   Nqs=None,qmin=None,qmax=None,\
                   demodulate=True,harmonic=3,response_power=-.5,\
                   localization=2,\
                   qscale='log',
                   **rpkwargs):
        """Tip radius and z in units of [nm], qmin and qmax in units of 1/a"""
        
        if verbose: Logger.write('Using Extended Monopole Model with parameters:\n'+\
                     '\tresponse magnitude: %s\n'%cls.response+\
                     '\tresponse exponent: %s\n'%response_power+\
                     '\tcharge localization: a/%s'%localization)
        
        #Needs log-distributed q-values for low-q weighted kernels GTop and GBottom
        rp,freqs,qs,zs=cls.__condition_arguments__(freqs,rp,zs,\
                                                   Nzs=Nzs,zmin=zmin,amplitude=amplitude,\
                                                   Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                                   qscale=qscale,\
                                                   **rpkwargs)
    
        #gtop=cls.GTop(rp,freqs,qs,zs,\
        #              a=a,response_power=response_power)
        gtop=1
        gbottom=cls.GBottom(rp,freqs,qs,zs,\
                            a=a,response_power=response_power,\
                            localization=localization)
        
        cls.raw_signal=gtop/(1-cls.response*gbottom)
        if verbose: Logger.write('\tSignal shape: %s'%repr(cls.raw_signal.shape))
        
        if demodulate:
            signal=cls.__demodulate__(cls.raw_signal,harmonic=harmonic)
            if hasattr(freqs,'__len__'):
                #Turn to spectrum (but don't tell to FFT along any new axes)
                signal=Spectrum(signal,axes=[freqs],\
                                axis_names=['Frequency'],\
                                axis=None)
        else:
            if hasattr(freqs,'__len__'):
                signal=Spectrum(cls.raw_signal,axes=[zs,freqs],\
                                axis_names=['Z','Frequency'],\
                                axis=None)
            else:
                signal=AWA(cls.raw_signal,axes=[zs],\
                                axis_names=['Z'])
        
        return signal
    
    @classmethod
    def GTop(cls,rp,freqs,qs,zs,\
             a=25,response_power=-.5,\
             localization=2):
        
        if verbose: Logger.write('Computing top q-integral...')
        
        #Broadcast everything into grids, zs the first axis, qs last axis#
        qs=qs*1e-7 #qs from units of cm-1 to nm^-1
        zs_grid,freqs_grid,qs_grid=broadcast_items(zs,freqs,qs)
        if isinstance(rp,numpy.ndarray):
            if rp.ndim==1: rp.resize(rp.shape+(1,))
            zs_grid,rp_grid=broadcast_items(zs,rp)
        else: rp_grid=rp
        
        ##Which *kerneltop* is picked appears to not matter much.##
        #Technically the second is more accurate, but more complicated too.
        #1) This applies if we just consider the const part of the homogeneous charge dist
        kerneltop=numpy.exp(-2*qs_grid*zs_grid)
        #2) This applies if we consider also the localized part of the homogeneous charge dist
        #kerneltop=(1/(a*qs_grid)+1/(localization+a*qs_grid))\
        #          *qs_grid*numpy.exp(-2*qs_grid*zs_grid)
        dGdq=kerneltop*rp_grid
        
        #Integrate over q-vector
        if verbose: Logger.write('\tdGdq shape: %s'%repr(dGdq.shape))
        cls.gtop=simps(dGdq,x=qs,axis=-1)
        
        if verbose: Logger.write('\tDone.')
        
        return cls.gtop

    @classmethod
    def GBottom(cls,rp,freqs,qs,zs,\
                a=25,response_power=-.5,\
                localization=2):
        """Tip radius and z in units of [nm], qmin and qmax in units of 1/a"""
        
        if verbose: Logger.write('Computing bottom q-integral...')
        
        #Broadcast everything into grids, zs the first axis, qs last axis#
        qs=qs*1e-7 #qs in nm^-1
        zs_grid,freqs_grid,qs_grid=broadcast_items(zs,freqs,qs)
        if isinstance(rp,numpy.ndarray):
            if rp.ndim==1: rp.resize(rp.shape+(1,))
            zs_grid,rp_grid=broadcast_items(zs,rp)
        else: rp_grid=rp
        
        kernelbottom=a*(a*qs_grid)**response_power\
                  *(a*qs_grid)/(localization+a*qs_grid)\
                  *numpy.exp(-2*qs_grid*zs_grid)
        dGdq=kernelbottom*rp_grid
        
        #Integrate over q-vector
        if verbose: Logger.write('\tdGdq shape: %s'%repr(dGdq.shape))
        cls.gbottom=simps(dGdq,x=qs,axis=-1)
        
        if verbose: Logger.write('\tDone.')
        
        return cls.gbottom
    
ExtendedMonopoleModel=_ExtendedMonopoleModel_()

class _DipoleModel_(TipModel):
    
    response_phase=1
    
    @classmethod
    def get_signal(cls,freqs,rp,zs=None,\
                   a=25,zmin=10,amplitude=50,Nzs=None,\
                   Nqs=None,qmin=None,qmax=None,\
                   demodulate=True,harmonics=[0,1,2,3],\
                   bonus_alpha=1,\
                   qscale='log',\
                   **rpkwargs):
        """Tip radius and z in units of [nm], qmin and qmax in units of 1/a"""
        
        #Needs log-distributed q-values for low-q weighted kernels GTop and GBottom
        rp,freqs,qs,zs=cls.__condition_arguments__(freqs,rp,zs,\
                                                   Nzs=Nzs,a=a,zmin=zmin,amplitude=amplitude,\
                                                   Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                                   qscale=qscale,\
                                                   **rpkwargs) #qs, zs in units of nm
        
        g=cls.G(rp,freqs,qs,zs)
        
        raw_signals=1/(1-cls.response_phase*bonus_alpha*g)-1
        if verbose: Logger.write('\tSignal shape: %s'%repr(raw_signals.shape))
            
        #Flip raw signal to make z-axis first
        raw_signals=AWA(numpy.array(raw_signals),\
                       axes=[zs*a,freqs], axis_names=['Z [nm]','Frequency'])
        cls.raw_signal=raw_signals
        
        #Demodulate if requested#
        if demodulate:
            
            #Iterate over all desired harmonics#
            signals={'signals':raw_signals}
            for harmonic in harmonics:
                signal=cls.__demodulate__(raw_signals,harmonic=harmonic)
                #Turn to spectrum (but don't tell to FFT along any new axes)
                signal=Spectrum(signal,axes=[freqs],\
                                axis_names=['Frequency'],\
                                axis=None).squeeze()
                signals['signal_%i'%harmonic]=signal
            
            return signals
                
        else: return raw_signals.squeeze()
        
        return signal
    
    @classmethod
    def G(cls,rp,freqs,qs,zs):
        """Tip radius and z in units of [nm], qmin and qmax in units of 1/a"""
        
        if verbose: Logger.write('Computing bottom q-integral...')
        
        #Broadcast everything into grids, zs the first axis, qs last axis#
        zs_grid,freqs_grid,qs_grid=broadcast_items(zs,freqs,qs) #second axis will be frequency axis, others will be integrated
        if isinstance(rp,numpy.ndarray):
            if rp.ndim==1: rp_grid=rp.reshape((1,)+rp.shape+(1,)) #rp has frequency axis
            elif rp.ndim==2: rp_grid=rp.reshape((1,)+rp.shape) #rp has frequency + q axes
        else: rp_grid=rp
        
        kernel=qs_grid**2*numpy.exp(-2*qs_grid*zs_grid)
        dGdq=kernel*rp_grid
        
        cls.rp_grid=rp_grid
        cls.zs_grid=zs_grid
        cls.gs_grid=qs_grid
        cls.kernel=kernel
        
        if verbose: Logger.write('\tdGdq shape: %s'%repr(dGdq.shape))
        cls.g=simps(dGdq,x=qs,axis=-1)
        
        if verbose: Logger.write('\tDone.')
        
        return cls.g
    
DipoleModel=_DipoleModel_()

class _SSEQModel_(TipModel):
        
    #This response setting seems to work best for reducing
    #resonances in realm of agreement for 4H SiC broadband
    response=1.4#*numpy.exp(1j*numpy.pi/2.\
                #           *1/20.)
    response_power=-.5
    quasistatic_correction=False
        
    taper=20
        
    @classmethod
    def get_nzs(cls,Nzs=None,zmin=1,amplitude=80,a=20,zscale='log'):
        
        if verbose: Logger.write('Building z-values...')
        
        if not Nzs: Nzs=10
        
        if zscale=='log':
            log=numpy.log
            zs=numpy.logspace(log(zmin)/log(10.),\
                              log(zmin+2*amplitude)/log(10.),\
                              Nzs)
        else:
            Logger.write('\tUsing linear z-scale...')
            zs=numpy.linspace(zmin,zmin+2*amplitude,Nzs)
            
        if verbose: Logger.write('\tDone.')
            
        return zs/float(a)
    
    @classmethod
    def demodulate(cls,signal,Nts=None,harmonic=3):
        """Takes z-axis as first axis, frequency as final axis."""
        
        #Assume z-axis is first axis#
        zs=signal.axes[0]; zmin=zs.min(); zmax=zs.max(); A=(zmax-zmin)/2.
        
        if Nts is None: Nts=50
        Nts=numpy.max((Nts,len(zs)))
        if verbose: Logger.write('Demodulating via spline interpolation over %i approach points...'%Nts)
        if verbose: Logger.write('\tSignal shape: %s'%repr(signal.shape))
        
        #Interpolate at desired z-values#
        cls.ts=numpy.linspace(0,.5,Nts) #Need only integrate over half period
        new_zs=zmin+A*(1-numpy.cos(2*numpy.pi*cls.ts))
        new_signal=signal.interpolate_axis(new_zs,axis=0,kind='slinear')
        
        #Demodulate
        ts_shape=(len(cls.ts),)+(1,)*(signal.ndim-1)
        ts_grid=cls.ts.reshape(ts_shape)
        signal=simps(numpy.cos(2*numpy.pi*harmonic*ts_grid)*new_signal,\
                     x=cls.ts,\
                     axis=0)
        
        if verbose: Logger.write('\tDone.')
        
        return signal
        
    @classmethod
    def condition_arguments(cls,freqs,rp,zs,a,\
                                Nzs=None,zmin=1,amplitude=80,\
                                Nqs=None,qmin=None,qmax=None,\
                                quadrature=numrec.GL,\
                                zscale='log',\
                                **rpkwargs):

        if verbose: Logger.write('Conditioning arguments to model...')

        ##Get zs first##
        if zs is None:
            nzs=cls.get_nzs(Nzs,zmin,amplitude,a,zscale)
        elif verbose:
            nzs=zs/float(a)
            Logger.write('Using provided z-value(s).')
        
        ##Get qs, in units of 1/a##
        if not Nqs: Nqs=244
        if not qmin:
            ##If we use quasistatic correction, we have an explicit minimum q!!#
            if cls.quasistatic_correction:
                k=2*numpy.pi*numpy.min(freqs)
                qmin=4*k*(a*1e-7) #put into dimensionless units of 1/a
            ##Otherwise use small fraction of tip radius##
            else: qmin=1e-3
        if not qmax: qmax=numpy.inf
        qs,ws=numrec.GetQuadrature(N=Nqs,xmin=qmin,xmax=qmax,\
                                   quadrature=quadrature)
        
        ##Get rp##
        if hasattr(rp,'__call__'):
            rp=cls.compute_rp(rp,freqs,qs*1/(a*1e-7),**rpkwargs)
        else:
            Logger.raiseException('rp must be a number or a function!  Instead got type %s.'%type(rp),\
                                  unless=(isinstance(rp,numpy.ndarray) \
                                          or type(rp) in number_types),\
                                  exception=TypeError)
        
        #Save most recent arguments
        cls.a=a
        cls.freqs=freqs
        cls.nzs=nzs
        cls.qs=qs
        cls.ws=ws
        cls.rp=rp
        
        if verbose: Logger.write('\tDone.')
        
        return rp,freqs,qs,ws,nzs
        
    @classmethod
    def GeometricResponse(cls,q,\
                          response=1.4):
        
        Lambda0=cls.response; response_exp=cls.response_power
        
        if q is 0: return 1
        
        return Lambda0*q**response_exp
        
    @classmethod
    def Lambda_q_s(cls,s,q):
        """This is the laplace X-form of the expression in Eq. 32 of the EMM paper."""
        
        Lambda=cls.GeometricResponse(q)
        
        alpha=numpy.tan(cls.taper/180.*numpy.pi)
        
        #Apply modulation of charge distribution's period
        n=1
        
        #lambda_q_s=Lambda*(1/(2+s)-1/(2+2*s)+1/(2*n*q+s)-1/(2*n*q+2*s))
        lambda_q_s=-Lambda*(2/numpy.sqrt((2*n*q+s)**2+(s*alpha)**2)\
                            -1/numpy.sqrt((n*q+s)**2+(s*alpha)**2))  #Minus sign added as correction 11/9/11
        
        return lambda_q_s
        
    @classmethod
    def quasistatic_correction_factor(cls,freq,q):
        
        if not cls.quasistatic_correction: return 1
        
        k=numpy.complex(2*numpy.pi*freq*(cls.a*1e-7)) #put into dimensionless units
        
        return q/numpy.sqrt(q**2-k**2)
        
    @classmethod
    def GetGMatrixElements(cls,freq,nz,rp,\
                           s,q,\
                           recompute=True):
        
        if recompute:
            cls.lambda_q_s=cls.Lambda_q_s(s,q)
        
        correction=cls.quasistatic_correction_factor(freq,q)
        cls.g=-rp*q*correction*numpy.exp(-2*q*nz)*cls.lambda_q_s #Minus sign added as correction 11/9/11
        
        return cls.g
        
    @classmethod
    def get_raw_signal_at_freq_and_z(cls,freq,nz,rp,\
                                     qs,ws,a,\
                                     recompute=True):
        
        #If desired, recompute items that can be stored
        if recompute:
            cls.ws=ws
            cls.qs=qs
            
            cls.s_grid,cls.q_grid=broadcast_items(qs,qs) #s is row vector, q is column vector
            cls.lambda0_vec=numpy.matrix(cls.Lambda_q_s(s=qs,q=0)).T
            cls.W=numpy.matrix(numpy.diag(ws))
            cls.I=numpy.matrix(numpy.eye(len(qs)))
            
        #Compute all matrix elements that are different each time
        ss=qs
        G=numpy.matrix(cls.GetGMatrixElements(freq=freq,nz=nz,rp=rp,\
                                              s=cls.s_grid,q=cls.q_grid,\
                                              recompute=recompute))
        cls.G=G
        cls.rp=rp
        cls.nz=nz
        cls.freq=freq
        
        #Invert total matrix
        lambda_s_vec=numpy.array((cls.I-G*cls.W).getI()*cls.lambda0_vec).squeeze()
        cls.lambda_s_vec=lambda_s_vec
        
        pre_part=(rp*ss**(1+cls.response_power)\
                   *numpy.exp(-2*ss*nz)\
                   *cls.quasistatic_correction_factor(freq,ss))/ss**2
        
        return numpy.sum(ws*pre_part*lambda_s_vec)
        
    @classmethod
    def get_charge_distribution(cls,zs):
        
        ss=cls.qs
        ws=cls.ws
        rp=cls.rp
        freq=cls.freq
        a=cls.a
        z=cls.z
        lambda_s_vec=cls.lambda_s_vec
        
        sum=0
        for i in range(len(ss)):
            s=ss[i]
            w=ws[i]
            charge_element=lambda_s_vec[i]
            charge_strength=w*charge_element*\
                            s*numpy.exp(-2*s*z/float(a))*\
                            cls.GeometricResponse(s)
            charge_contribution=-(2*numpy.exp(-2*s*zs/float(a))-numpy.exp(-s*zs/float(a)))
            
            sum+=charge_strength*charge_contribution
        
        return AWA(sum,axes=[zs],axis_names=['Z'])
        
    @classmethod
    def get_q_distribution(cls,qs):
        
        ss=cls.qs
        ws=cls.ws
        rp=cls.rp
        freq=cls.freq
        a=cls.a
        z=cls.z
        lambda_s_vec=cls.lambda_s_vec
        
        sum=0
        q_contribution_correction=cls.quasistatic_correction_factor(freq,qs)
        alpha=numpy.tan(numpy.pi*cls.taper/180.)
        
        for i in range(len(ss)):
            s=ss[i]
            w=ws[i]
            charge_element=lambda_s_vec[i]
            charge_strength=w*charge_element*\
                            s*numpy.exp(-2*s*z/float(a))*\
                            cls.GeometricResponse(s)
            q_contribution=-qs*(2/numpy.sqrt((qs+2*s)**2+(qs*alpha)**2)+\
                                    -1/numpy.sqrt((qs+s)**2+(qs*alpha)**2))/2.
            
            sum+=charge_strength*q_contribution
        
        return AWA(sum*numpy.exp(-qs*z),axes=[qs],axis_names=['q-vector'])
        
    @classmethod
    def get_signal(cls,freqs,rp,zs=None,\
                   a=25,zmin=1,amplitude=80,Nzs=15,\
                   Nqs=72,qmin=None,qmax=None,\
                   quadrature=numrec.GL,\
                   demodulate=True,Nts=50,harmonics=[1,2,3],\
                   zscale='log',\
                   **rpkwargs):
        """qmin and qmax in units of 1/a"""
        
        if verbose: Logger.write('Using SSEQ Model with parameters:\n'+\
                     '\tresponse magnitude: %s\n'%cls.response+\
                     '\tresponse exponent: %s\n'%cls.response_power)
        
        #Make sure frequencies are iterable
        if not hasattr(freqs,'__len__'): freqs=[freqs]
        freqs=numpy.array(freqs)
        
        #Get quadrature q values and heights z
        rp,freqs,qs,ws,nzs=cls.condition_arguments(freqs,rp,zs,a,\
                                                      Nzs=Nzs,zmin=zmin,amplitude=amplitude,\
                                                      Nqs=Nqs,qmin=qmin,qmax=qmax,\
                                                      quadrature=quadrature,\
                                                      zscale=zscale,\
                                                      **rpkwargs)
        
        #Compute raw signal at each frequency, z-height
        raw_signals=[]
        recompute=True #At first, want to recompute all charge dists etc.
        for i,freq in enumerate(freqs):
            raw_signal_at_freq=[]
            progress=i*len(nzs)/float(len(nzs)*len(freqs))*100.
            if verbose: Logger.write('\tPROGRESS: %i%% - Computing raw signal at frequency w=%scm^-1...'%\
                         (progress,freq))
            
            if isinstance(rp,AWA): rp_at_freq=rp.cslice[freq]
            else: rp_at_freq=rp
            
            for j,nz in enumerate(nzs):
                raw_signal_at_freq.append(cls.get_raw_signal_at_freq_and_z(freq=freq,nz=nz,rp=rp_at_freq,\
                                                                           qs=qs,ws=ws,a=a,\
                                                                           recompute=recompute))
                recompute=True
                
            raw_signals.append(raw_signal_at_freq)
        
        #Flip raw signal to make z-axis first
        raw_signals=AWA(numpy.array(raw_signals).transpose(),\
                       axes=[nzs*a,freqs], axis_names=['Z','Frequency']) #z axis in units of a
        cls.raw_signals=raw_signals
        
        if verbose: Logger.write('\tSignal shape: %s'%repr(cls.raw_signals.shape))
        
        #Demodulate if requested#
        if demodulate:
            
            #Iterate over all desired harmonics#
            signals={}
            for harmonic in harmonics:
                signal=cls.demodulate(raw_signals,Nts=Nts,harmonic=harmonic)
                #Turn to spectrum (but don't tell to FFT along any new axes)
                signal=Spectrum(signal,axes=[freqs],\
                                axis_names=['Frequency'],\
                                axis=None).squeeze()
                signals[harmonic]=signal
            
            return signals
                
        else:
            return raw_signals.squeeze()
    
SSEQModel=_SSEQModel_()

def get_charge_data_path(geometry,L,skin_depth,taper_angle,quadrature_type,Nzs,Nqs,freq):
    
    if geometry=='sphere': L=2

    geometry_title=geometry.capitalize()
    if geometry in ('cone','hyperboloid'):
        filepath=os.path.join(charge_data_dir,\
                              '%sCharge_L=%.2E_SkinDepth=%.2E_Taper=%i_Quad=%s_Nzs=%i_Nqs=%i_freq=%.2E.pickle'%\
                              (geometry_title,L,skin_depth,taper_angle,quadrature_type,Nzs,Nqs,freq))
    elif geometry=='PtSi': #Taper angle is disabled
        filepath=os.path.join(charge_data_dir,\
                              '%sCharge_L=%.2E_SkinDepth=%.2E_Quad=%s_Nzs=%i_Nqs=%i_freq=%.2E.pickle'%\
                              (geometry_title,L,skin_depth,quadrature_type,Nzs,Nqs,freq))
    else:
        filepath=os.path.join(charge_data_dir,\
                              '%sCharge_L=%.2E_SkinDepth=%.2E_Quad=%s_Nzs=%i_Nqs=%i_freq=%.2E.pickle'%\
                              (geometry_title,L,skin_depth,quadrature_type,Nzs,Nqs,freq))
            
    return filepath

class _LightningRodModel_(TipModel):
        
    #Hyperboloid geometry tends to predict stronger material contrast
    geometric_params={'a':30,\
                      'L':19000/30.,\
                      'skin_depth':0.05,\
                      'taper_angle':20,\
                      'geometry':'hyperboloid',\
                      'beam_shape':'plane_wave',\
                      'incidence_angle':30}
    
    load_params={'reload_model':True,\
                 'quadrature':quadrature,\
                 'Nzs':244,\
                 'Nqs':244,\
                 'freq':30e-7*1000,\
                 'comsol_lambda0':False,\
                 'comsol_filename':'Comsol_AvgCharge_60deg_WL10um.pickle'}
    
    quadrature_params={'xWarp':True,\
                       'quadrature':quadrature,\
                       'x0':.99,\
                       'b':.75,\
                       'q_correction':False,\
                       'q_correction_exponent':1,\
                       'interpolation':'linear'} #This b-value obtains convergence for both SiC and SiO2 at Nqs>=144
    
    resonant_sample=True #Internal self consistency with calculated charge response requires this to be True
    
    def __call__(self,*args,**kwargs):
        
        ##Make sure all "ambient" arguments are identified
        if 'ambient_rp' not in kwargs: kwargs['ambient_rp']=None
        if 'normalization_ambient_rp' not in kwargs: kwargs['normalization_ambient_rp']=None
        ambient_rp=kwargs['ambient_rp']
        normalization_ambient_rp=kwargs.pop('normalization_ambient_rp')
        
        ##See if we'll want to normalize##
        exkwargs=misc.extract_kwargs(kwargs,\
                                     normalize_to=None,normalize_at=None,\
                                     normalization_ambient=None)
        normalize_to=exkwargs['normalize_to']
        normalize_at=exkwargs['normalize_at']
        if 'demodulate' in kwargs: demodulate=kwargs['demodulate']
        else: demodulate=True
        
        ##Get signal using whatever model##
        if verbose: Logger.write('Getting signal...')
        signal=self.get_signal(*args,**kwargs)
        if verbose: Logger.write('Done getting signal')
        
        if normalize_to and demodulate:
            if verbose: Logger.write('Getting normalization signal...')
            
            original_reload_model=self.load_params['reload_model']
            self.load_params['reload_model']=False
            kwargs['rp']=normalize_to
            
            ##Pick local ambient for normalization if we must pick something#
            if ambient_rp and not normalization_ambient_rp: 
                Logger.write('\tUsing same ambient rp as that for the sample spectrum (quasi-local normalization)...')
                normalization_ambient_rp=normalize_to
            kwargs['ambient_rp']=normalization_ambient_rp
            
            if normalize_at is not None:
                if len(args): args=list(args); args[0]=normalize_at
                else: kwargs['freqs']=normalize_at
            normalization=self.get_signal(*args,**kwargs)
            
            #Take note that signal will be a dictionary of harmonics
            for key in list(signal.keys()):
                signal['sample_'+key]=signal[key]
                signal['norm_'+key]=normalization[key]
                #if signal is a harmonic, normalize
                if re.compile('signal_[0-9]+').search(key): 
                    if signal[key].ndim != normalization[key].ndim:
                        normalization[key].resize(normalization[key].shape+(1,))
                    signal[key]=signal[key]/normalization[key]
            
            if verbose: Logger.write('Done normalizing.')
            self.load_params['reload_model']=original_reload_model
        
        return signal
        
    
    def get_zs(self,Nzs=None,zmin=1,amplitude=80):
        
        if not Nzs: Nzs=20
        
        a=self.geometric_params['a']
        zmin/=float(a)
        zmax=zmin+(2*amplitude)/float(a)
        
        log=numpy.log
        zs=numpy.logspace(log(zmin)/log(10.),\
                          log(zmax)/log(10.),\
                          Nzs)
            
        return zs
    
    @staticmethod
    def demodulate(signals,harmonics=list(range(5)),Nts=None,\
                   quadrature=numrec.GL):
        """Takes z-axis as first axis, frequency as final axis."""
    
        global ts,wts,weights,signals_vs_time,zs
    
        #max harmonic resolvable will be frequency = 1/dt = Nts
        if not Nts: Nts=4*numpy.max(harmonics)
        if isinstance(quadrature,str) or hasattr(quadrature,'calc_nodes'):
            ts,wts=numrec.GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
            
        else:
            ts,wts=numpy.linspace(-.5,0,Nts),None
            if quadrature is None: quadrature=simps
        
        freqs=signals.axes[1]
        zmin=signals.axes[0].min()
        zmax=signals.axes[0].max()
        amplitude=(zmax-zmin)/2.
        zs=zmin+amplitude*(1+numpy.cos(2*numpy.pi*ts))
        
        harmonics=numpy.array(harmonics).reshape((len(harmonics),1))
        weights=numpy.cos(2*numpy.pi*harmonics*ts)*wts
        weights_grid=weights.reshape(weights.shape+(1,)*(signals.ndim-1))
        
        signals_vs_time=signals.interpolate_axis(zs,axis=0) ; signals_vs_time.set_axes([ts],axis_names=['t'])
        
        if wts is not None:
            demodulated=2*2*numpy.sum(signals_vs_time*weights_grid,axis=1) #perform quadrature
        else: demodulated=2*2*quadrature(signals_vs_time,x=ts,axis=1)
    
        demodulated=Spectrum(demodulated,axes=[harmonics,freqs],axis_names=['harmonic','Frequency'])
        
        return signals_vs_time,demodulated
    
    #Something is wrong with this demodulate function, not sure what, remains unresolved.
    #Results disagree with the simpler `demodulate(...)`
    def demodulate2(self,signals,Nts=None,harmonics=[1,2,3]):
        """Takes z-axis as first axis, frequency as final axis."""
        
        #Assume z-axis is first axis#
        zs_nm,freqs=signals.axes
        zmin=zs_nm.min()
        zmax=zs_nm.max()
        A=(zmax-zmin)/2.
        
        #Decide time values and new z values
        if Nts is None: Nts=500
        Nts=numpy.max((Nts,len(zs_nm)))
        ts=numpy.arange(Nts)/numpy.float(Nts)*.5#snumpy.linspace(0,.5,Nts) #Need only integrate over half period
        new_zs=zmin+A*(1-numpy.cos(2*numpy.pi*ts))
        ts_grid=ts.reshape((Nts,1))
        self.ts=ts
        
        #Interpolate
        if verbose: Logger.write('Demodulating at desired harmonics...'); time1=time.time()
        #try:
        #    interp1=RectBivariateSpline(x=zs_nm,y=freqs,z=signals.real,s=0)
        #    interp2=RectBivariateSpline(x=zs_nm,y=freqs,z=signals.imag,s=0)
        #    new_signals=interp1(x=new_zs,y=freqs)\
        #                 +1j*interp2(x=new_zs,y=freqs)
        #    if verbose:
        #        Logger.write('\tInterpolated with bivariate spline, time: %1.2f'%(time.time()-time1))
        #    
        #In case RectBivariateSpline needs more points to run
        #except:
        #    new_signals=[]
        #    for i in xrange(len(freqs)):
        #        interp1=UnivariateSpline(x=zs_nm,y=signals[:,i].real,s=0)
        #        interp2=UnivariateSpline(x=zs_nm,y=signals[:,i].imag,s=0)
        #        new_signals.append(interp1(new_zs)+1j*interp2(new_zs))
        #    new_signals=numpy.array(new_signals).transpose()
        #    if verbose:
        #        Logger.write('\tInterpolated with sequence of univariate splines, time: %1.2f'%(time.time()-time1))
                
        new_signals=signals.interpolate_axis(new_zs,axis=0)
        signal_v_time=AWA(new_signals,axes=[ts,freqs],axis_names=['T','Frequency'])
        
        #Demodulate
        demodulated_signals={}
        for harmonic in harmonics:
            demodulated_signal=2*simps(numpy.cos(2*numpy.pi*harmonic*ts_grid)*signal_v_time,\
                                                 x=ts,axis=0)*2 #Last factor of 2 to make up for integrating only half period
            demodulated_signal=Spectrum(demodulated_signal,axes=[freqs],axis_names=['Frequency'],axis=None)
            demodulated_signals[harmonic]=demodulated_signal
        
        if verbose: Logger.write('\tDone.')
        
        return signal_v_time.squeeze(),demodulated_signals
    
    
    def load_comsol_lambda0(self,filename):
        
        qs,zs=self.charges.axes
        Rs=self.charge_radii
        freq=self.load_params['freq']
        
        Logger.write('Loading electrodynamic charge data from file "%s"...'%filename)
        try: file=open(os.path.join(charge_data_dir,filename))
        except IOError: Logger.raiseException('No pre-computed Lambda0 data was found for parameters:\n'+\
                                              'Nqs=%i\n'%Nqs)
        
        from common import unpickle_legacy
        self.charges0=unpickle_legacy(filename)
        
        file.close()
        
        zs0=self.charges0.axes[0]
        zs0-=zs0.min()
        Rs0=AWA(Rs,axes=[zs]).interpolate_axis(zs0,axis=0,extrapolate=True,bounds_error=False)
        
        charge_grid=self.charges0.reshape((len(zs0),1))
        zs_grid=zs0.reshape((len(zs0),1))
        Rs_grid=Rs0.reshape((len(Rs0),1))
        ss_grid=qs.reshape((1,len(qs)))
        k=2*numpy.pi*freq
        
        #pref_grid=numpy.exp(-ss_grid*zs_grid)*j0(numpy.sqrt(ss_grid**2+k**2)*Rs_grid)
        
        skin_depth=self.geometric_params['skin_depth']
        if skin_depth:
            factor=1#-1j
            delta=skin_depth/factor
            pref_grid=(numpy.exp(-ss_grid*zs_grid)-numpy.exp(-zs_grid/delta))\
                      /(1-ss_grid*delta)\
                     *j0(numpy.sqrt(ss_grid**2+k**2)*Rs_grid)
        else:
            pref_grid=numpy.exp(-ss_grid*zs_grid)\
                      *j0(numpy.sqrt(ss_grid**2+k**2)*Rs_grid)
        
        integrand=pref_grid*charge_grid
        self.Lambda0=AWA(simps(x=zs0,y=integrand,axis=0),\
                        axes=[qs],axis_names=['s'])
    
    
    def load_charge_data(self):
        
        #All the stored geometry and quadrature parameters determine
        #which charge data to load
        geometry=self.geometric_params['geometry']
        L=self.geometric_params['L']
        skin_depth=self.geometric_params['skin_depth']
        taper=self.geometric_params['taper_angle']
        Nzs=self.load_params['Nzs']
        Nqs=self.load_params['Nqs']
        freq=self.load_params['freq']
        quadrature_type=self.load_params['quadrature']
        
        filepath=get_charge_data_path(geometry,L,skin_depth,taper_angle,quadrature_type,Nzs,Nqs,freq)
                     
        if verbose: Logger.write('Loading charge data from file "%s"...'%filepath)
        try: file=open(filepath,'rb')
        except IOError:
            Logger.raiseException('No pre-computed charge data was found correspondent '+\
                                  'to the desired charge profile:\n'+\
                                  '"%s"'%filepath)
        from common.misc import unpickle_legacy
        charge_data=unpickle_legacy(filepath)
        
        self.qs,self.wqs=charge_data['quadrature'] #qs, ws
        self.charges=charge_data['charges'] #axes s, z x q
        self.Lambda=charge_data['integral_xforms'].transpose() #axes q, s --> s, q
        self.charge_data=charge_data
        self.dipole_moments=charge_data['dipole_moments'] #array with axis s
        
        self.charge_radii=charge_data['charge_radii']
        if self.charge_radii.ndim is 2: self.charge_radii=self.charge_radii[0]
        
        try: self.charge_quadrature=charge_data['charge_quadrature']
        except KeyError: pass
        
        if self.load_params['comsol_lambda0']:
            filename=self.load_params['comsol_filename']
            self.load_comsol_lambda0(filename)
            
        else:
            beam_shape=self.geometric_params['beam_shape']
            incidence_angle=self.geometric_params['incidence_angle']
            
            if verbose: Logger.write('\tUsing incident beam profile: "%s"'%beam_shape+\
                                     '\n\tIncidence angle: %s degrees'%incidence_angle)
            
            ##Try to load with specified incidence angle
            try:
                self.Lambda0=charge_data['integral_xforms_%s%s'%(beam_shape,incidence_angle)]
                self.charges0=charge_data['charges_%s%s'%(beam_shape,incidence_angle)]
                
                self.Lambda0Refl=charge_data['integral_xforms_%s%s'%(beam_shape,180-incidence_angle)]
                self.charges0Refl=charge_data['charges_%s%s'%(beam_shape,180-incidence_angle)]
                
            except KeyError:
                
                self.Lambda0=charge_data['integral_xforms_%s'%beam_shape]
                self.charges0=charge_data['charges_%s'%beam_shape]
                
                self.Lambda0Refl=charge_data['integral_xforms_%s'%beam_shape]
                self.charges0Refl=charge_data['charges_%s'%beam_shape]
        
    
    def prepare_model(self,zs=None,zmin=1e-1,Nzs=None,Nqs=122,amplitude=80,a=None,interpolation=None,**kwargs):

        if verbose: Logger.write('Preparing model...')
    
        ##Get zs first##
        if a: self.geometric_params['a']=a
        if zs is None: zs=self.get_zs(Nzs,zmin,amplitude)
        else:
            if verbose: Logger.write('Using provided z-value(s).')
            if not hasattr(zs,'__len__'): zs=[zs]
            zs=numpy.array(zs).astype(float)
            zs/=self.geometric_params['a'] #take incoming nm values and normalize to tip radius
        self.zs=zs
        
        #Load/modify all the next stuff only if reloading model
        if self.load_params['reload_model']:
        
            for key in list(kwargs.keys()):
                if key.startswith('load_'):
                    new_key=key[len('load_'):]
                    self.load_params[new_key]=kwargs.pop(key)
            
            #Store all the provided geometry and quadrature parameters
            if interpolation: self.quadrature_params['interpolation']=interpolation
            if 'geometry' in kwargs: self.geometric_params['geometry']=kwargs.pop('geometry')
            if 'taper_angle' in kwargs: self.geometric_params['taper_angle']=kwargs.pop('taper_angle')
            
            ##Load charge data
            self.load_charge_data()
        
        xWarp=self.quadrature_params['xWarp']
        if xWarp:
            if verbose: Logger.write('\tComputing xWarp quadrature for q-values...')
            self.get_xWarp_coords(qmin=self.qs.min(),qmax=20,Nqs=Nqs)
        else:
            if verbose: Logger.write('\tUsing the already loaded quadrature rather than xWarp quadrature...')
            self.qxs,self.wqxs=self.qs,self.wqs
        
    
    def get_xWarp_coords(self,qmin,qmax,Nqs):
        
        quadrature=self.quadrature_params['quadrature']
        if quadrature=='TS': quadrature=numrec.TS
        elif quadrature=='GL': quadrature=numrec.GL
        elif quadrature=='CC': quadrature=numrec.CC
        
        xs,wxs=numrec.GetQuadrature(xmin=-1,xmax=1,N=Nqs,quadrature=quadrature)
        
        #x0=self.quadrature_params['x0']
        b=self.quadrature_params['b']
        x0=xs[-2]
        
        a=(2*b+x0)*((1+x0)/(1-x0))**-b\
          /float(2*b/1e2) #focus on q0=1e2/a
        q1=qmin; deltaq=qmax-q1; delta=2*(deltaq/a)**(-1/float(b))
        qxs=a*((1+xs)/(1-xs+delta))**b+q1
        
        #New weights results from change of variables: dqx = dqx/dx * dx
        wqxs=a*b*((1+xs)/(2+delta))**b\
                *((2+delta)/(1-xs+delta))**(1+b)\
                /(1+xs)*wxs
        
        self.qxs=qxs
        self.wqxs=wqxs
        
        return qxs,wqxs
        
    
    def Lambda0Vector(self,qxs):
        
        Lambda0=self.Lambda0.interpolate_axis(qxs,axis=0,extrapolate=True,bounds_error=False)
        
        return numpy.matrix(Lambda0).T
    
    def Lambda0VectorRefl(self,qxs):
        
        Lambda0Refl=self.Lambda0Refl.interpolate_axis(qxs,axis=0,extrapolate=True,bounds_error=False)
        
        return numpy.matrix(Lambda0Refl).T
    
    def LambdaMatrix(self,qxs):
        
        #Should be no bounds error, we only interpolate into sampled region
        Lambda=self.Lambda.interpolate_axis(qxs,axis=0,kind='linear',bounds_error=False)\
                         .interpolate_axis(qxs,axis=1,kind='linear',bounds_error=False)
                         
        return numpy.matrix(Lambda)
        
    
    def evaluate_rp(self,freq,rp,qxs,**rpkwargs):
        
        a=self.geometric_params['a'] #used to convert qxs from units of 1/a
        k=2*numpy.pi*freq
        
        #exclude near the light line as comparatively unimportant
        if self.resonant_sample: Qs=numpy.sqrt(k**2+(qxs/(a*1e-7))**2) #This is technically the right way to evaluate either way, since qxs correspond to out-of-plane propogation
        else: Qs=qxs/(a*1e-7)
        
        #Evaluate surface response#
        if hasattr(rp,'__call__'):
            #evaluate with q in units of cm-1 rather than 1/a
            rp=rp(freq, Qs, **rpkwargs)
        elif isinstance(rp,AWA) and 'Frequency' in rp.axis_names[0]:
            rp=rp.cslice[freq]
        
        self.rp=rp
        
        if isinstance(rp,numpy.ndarray):
            Logger.raiseException('If "rp" evaluates to an array, it must be of dimension 0, 1 or 2.',\
                                   unless=(rp.ndim in [0,1,2]), exception=ValueError)
            #G matrix evaluation will take care of turning rp into a matrix
        else:
            Logger.raiseException('"rp" must evaluate to a scalar or a 0-, 1- or 2-dimensional ndarray.',\
                                  unless=(type(rp) in number_types), exception=ValueError)
        
        return rp
        
    
    def GMatrix(self,z,freq,rp):
        
        a=self.geometric_params['a']
        knorm=2*numpy.pi*a*1e-7*freq
        
        if isinstance(rp,numpy.ndarray) and rp.ndim is 2:
            #Reflection matrix integral operator - it's not responsible for including its own weights (how would it know?)
            R=numpy.matrix(rp)
            #Propagator
            P=numpy.matrix(numpy.diag(numpy.exp(-self.qxs*z)))
            #Distribution of momenta
            D=numpy.matrix(numpy.diag(self.qxs))
            #If R is diagonal, then everything will be diagonal and will mutually commute
            W=numpy.matrix(numpy.diag(self.wqxs))
            
            return -D*P*W*R*P #note negative sign, to maintain typical notion of G in scattering problem
        
        else:
            pref=self.qxs
            #pref=(self.qxs**2+knorm**2)**(3/2.)/self.qxs**2 #; print 'Full calculation!' #Works for SiO2, less so for SiC
            return -numpy.matrix(numpy.diag(pref\
                                            *numpy.exp(-2*self.qxs*z)\
                                            *rp))
        
    
    def compute_state(self,Lambda0Vecx,LambdaMatx,\
                          z,freq,rp):
        
        I=numpy.matrix(numpy.eye(len(Lambda0Vecx)))
        WMatx=numpy.matrix(numpy.diag(self.wqxs))
        GMatx=self.GMatrix(z,freq,rp)
        
        #Lambda stuff should already be global, passed down from above
        self.WMatx=WMatx
        self.GMatx=GMatx
        
        #Compute state psi along xWrap q-coordinates
        kernelx=LambdaMatx*WMatx*GMatx
        soln=solve((I-kernelx),Lambda0Vecx)
        psi_qxs=numpy.array(GMatx*soln).squeeze()
        psi_qxs=AWA(psi_qxs,axes=[self.qxs],axis_names=['s'])
        self.z=z
        self.freq=freq
        self.psi=psi_qxs
        self.kernel=kernelx
        
        return psi_qxs
        
    def get_dipole_moments(self,qs):
        
        theta=self.geometric_params['incidence_angle']
        if isinstance(self.dipole_moments,dict):
            pzs=self.dipole_moments[theta].interpolate_axis(qs,axis=-1)
        else: pzs=self.dipole_moments.interpolate_axis(qs,axis=-1)
        
        if self.quadrature_params['q_correction']:
            pzs*=qs**self.quadrature_params['q_correction_exponent'] #Exponent here does not much matter *phew*
        
        #return a 2-D array
        return pzs
    
    def get_dipole_moments_refl(self,qs):
        
        theta=180-self.geometric_params['incidence_angle']
        if isinstance(self.dipole_moments,dict):
            pzs=self.dipole_moments[theta].interpolate_axis(qs,axis=-1)
        else: pzs=self.dipole_moments.interpolate_axis(qs,axis=-1)
        
        if self.quadrature_params['q_correction']:
            pzs*=qs**self.quadrature_params['q_correction_exponent'] #Exponent here does not much matter *phew*
        
        #return a 2-D array
        return pzs
    
    def get_charge_distribution(self,zs=None,psi=None):
        "Provide zs in nm."
        
        qxs=self.qxs
        wqxs=self.wqxs
        a=self.geometric_params['a']
        
        if psi is None: 
            try: psi=self.psi
            except AttributeError:
                Logger.raiseException('The latest tip state is empty! '+\
                                      'Please call `LightningRodModel.get_signal(...)` first.')
        
        #iterate through all q values and weight charges accordingly
        charge=0
        for i,qx in enumerate(qxs):
            #Make sure we weight by wqxs
            charge=charge+psi[i]*self.charges.interpolate_axis(qx,axis=0)*wqxs[i]
            
        zs_norm=charge.axes[0]
        charge=AWA(charge,axes=[zs_norm*a],axis_names=['Z [nm]'])
        
        #Evaluate at specific z-values (in nm), if requested
        if zs is not None:
            charge=charge.interpolate_axis(zs,axis=-1,\
                                           extrapolate=False,bounds_error=False,\
                                           fill_value=0)
            
        return charge
    
    def get_radiation_pattern(self,thetas=numpy.linspace(0,360,500),\
                              psi=None,freq=None,\
                              plot=False,ax=None):
        
        try: zs,wzs=self.charge_quadrature
        except AttributeError:
            zs=self.charges0.axes[0]
            wzs=None
            
        Rs=self.charge_radii
        
        if freq is None:
            try: freq=self.freq
            except AttributeError:
                Logger.raiseException('The latest frequency value is empty! '+\
                                      'Please call `LightningRodModel.get_signal(...)` first.')
        
        a=self.geometric_params['a']
        RadGen=RadPerChargeRingGenerator(zs,Rs,freq=freq*a*1e-7,wzs=wzs)
        
        if psi is None:
            try: psi=self.psi
            except AttributeError:
                Logger.raiseException('The latest tip state is empty! '+\
                                      'Please call `LightningRodModel.get_signal(...)` first.')
                
        charge=self.get_charge_distribution(zs=zs*a,psi=psi)
        rad=RadGen(charge,thetas)
        
        if plot and hasattr(thetas,'__len__'):
            from matplotlib import pyplot as plt
            if ax is None: plt.figure()
            else: plt.sca(ax)
            absrad=numpy.abs(rad)
            absrad=absrad[numpy.isfinite(absrad)]
            plt.polar(absrad.axes[0]-numpy.pi/2.,absrad/absrad.max())
            #plt.gca().set_theta_direction(-1)
            plt.yticks([]);plt.ylim(0,1.1)
            theta_ticks=numpy.array([0,30,45,60,90,120,135,150,\
                                     180,210,225,240,270,300,315,330])
            plt.xticks((theta_ticks)*numpy.pi/180.,\
                       theta_ticks)
            plt.tight_layout()
            plt.grid(color='k',ls='--')
            
        return rad
    
    def get_momentum_distribution(self,psi=None):
        
        qxs=self.qxs
        wqxs=self.wqxs
        
        if psi is None: psi=self.psi
        psi_qxs_vec=numpy.matrix(psi).T
        LambdaMat=self.LambdaMatrix(qxs)
        WMat=numpy.matrix(numpy.diag(wqxs))
        QMat=numpy.matrix(numpy.diag(qxs))
        field_contributions=numpy.array(QMat*LambdaMat*WMat*psi_qxs_vec).squeeze()
        
        return AWA(field_contributions,axes=[qxs],axis_names=['q'])
        
    
    def get_field_distribution(self,zs,rs,freq,material):
        
        eps1=1 #air
        from NearFieldOptics import Materials as mat
        if isinstance(material,mat.Material): eps2=material.epsilon(freq)
        elif isinstance(material,mat.LayeredMedia): eps2=material.get_exit().epsilon(freq)
        rp=material.reflection_p
        
        d=self.z
        a=self.geometric_params['a']
        zs=zs[zs<=d*a]
        Logger.raiseException('This evaluator is only applicable to values z<=d, '+\
                              'but no such values have been provided!',\
                              unless=len(zs), exception=ValueError)
        
        qxs=self.qxs
        wqxs=self.wqxs
        
        psi_qxs_vec=numpy.matrix(self.psi).T
        LambdaMat=self.LambdaMatrix(qxs)
        Lambda0Vec=self.Lambda0Vector(qxs)
        WMat=numpy.matrix(numpy.diag(wqxs))
        QMat=numpy.matrix(numpy.diag(qxs))
        #field_contributions=qxs**2*numpy.exp(-qxs)
        field_contributions=numpy.array(QMat*(Lambda0Vec+LambdaMat*WMat*psi_qxs_vec)).squeeze()
        
        #Add in homogeneous field contribution
        self.field_contributions=AWA(field_contributions,axes=[qxs],axis_names=['q'])
        
        zs_norm=zs/float(a)
        rs_norm=rs/float(a)
        rs_norm,zs_norm=broadcast_items(rs_norm,zs_norm)
        #print zs_norm.max(),zs_norm.min(),d
        
        k=2*numpy.pi*freq*(a*1e-7) #put into dimensionless number
        total_zfield=0
        total_rfield=0
        for field_contrib,q,wq in zip(field_contributions,qxs,wqxs):
            
            Q=numpy.sqrt(k**2+q**2)
            rp_value=rp(freq,Q/(a*1e-7))
            
            #The transmitted field is not usefully defined in terms of
            #transmission-p coefficient, field orientations relative to
            #a diffracted Poynting vector are ill-defined for an evanescent field
            T_r=1-rp_value
            T_z=eps1/eps2*(1+rp_value)
            
            zpos_dependence1=(numpy.exp(q*(zs_norm-d))\
                              +rp_value*numpy.exp(-q*(d+zs_norm)))*j0(Q*rs_norm)
            zpos_dependence2=T_z*numpy.exp(q*(zs_norm-d))*j0(Q*rs_norm)
            zpos_dependence=numpy.where(zs_norm>=0,zpos_dependence1,zpos_dependence2)
            self.zpos_dependence=zpos_dependence
            
            rpos_dependence1=(numpy.exp(q*(zs_norm-d))\
                              -rp_value*numpy.exp(-q*(d+zs_norm)))*j1(Q*rs_norm)
            rpos_dependence2=T_r*numpy.exp(q*(zs_norm-d))*j1(Q*rs_norm)
            rpos_dependence=numpy.where(zs_norm>=0,rpos_dependence1,rpos_dependence2)
            
            total_zfield+=(-field_contrib)*zpos_dependence*wq
            total_rfield+=(-field_contrib)*rpos_dependence*wq
            
        total_zfield=AWA(total_zfield,axes=[rs,zs],axis_names=['r','z'])
        total_rfield=AWA(total_rfield,axes=[rs,zs],axis_names=['r','z'])
        
        return total_zfield,total_rfield
        
    
    def get_signal(self,freqs,rp,ambient_rp=None,zs=None,Nzs=20,zmin=1e-1,amplitude=80,\
                   a=None,interpolation=None,\
                   Nqs=122,demodulate=True,Nts=None,harmonics=[0,1,2,3,4],\
                   **kwargs):
        """qmin and qmax in units of 1/a
        With TS quadrature, Nqs=122 or 244 are appropriate values
        to ensure convergence."""
        
        if verbose:
            Logger.write('Computing near-field signal with Lightning Rod Model...')
        
        #Make sure frequencies are iterable
        if not hasattr(freqs,'__len__'): freqs=[freqs]
        freqs=numpy.array(freqs)
        
        #Prepare the model#
        self.prepare_model(zs=zs,zmin=zmin,Nzs=Nzs,Nqs=Nqs,amplitude=amplitude,a=a,kwargs=kwargs)
                
        #Compute necessary matrices on our q-quadrature grid
        self.Lambda0Vecx=self.Lambda0Vector(self.qxs)
        self.LambdaMatx=self.LambdaMatrix(self.qxs)
        
        ##In case there's an ambient rp##
        if ambient_rp is not None:
            ambient_rp=ambient_rp(freqs,angle=self.geometric_params['incidence_angle'])
        
        #Compute psi v. (freq, q) at each z
        psis=[]
        for i,freq in enumerate(freqs):
            progress=i/float(len(freqs))*100.
            if verbose: Logger.write('\tPROGRESS: %i%% - Computing state of the tip charge v. '%progress+\
                                     'z & s at freq=%s cm^-1...'%freq)
            
            ##In case there's an ambient rp, update the initial charge distribution##
            if ambient_rp is not None:
                Lambda0Vecx=self.Lambda0Vecx+ambient_rp.cslice[freq]*self.Lambda0VectorRefl(self.qxs)
            else: Lambda0Vecx=self.Lambda0Vecx
            
            #Get distribution of charge w.r.t. original q-values
            rp_at_freq=self.evaluate_rp(freq,rp,self.qxs,**kwargs)
            psis_at_freq=[self.compute_state(Lambda0Vecx,self.LambdaMatx,\
                                            z,freq,rp=rp_at_freq) \
                                for z in self.zs[::-1]][::-1]
            psis.append(psis_at_freq)
            
        #Make into master array
        a=self.geometric_params['a']
        zs_nm=self.zs*a
        psis=Spectrum(psis,axes=[freqs,zs_nm,self.qxs],\
                      axis_names=['Frequency','Z [nm]','q [1/a]'],axis=None) #don't tell to FFT along any new axes!
        
        #Permute the axes to make zs first dimension (for convenience)
        psis=psis.transpose((1,0,2)) #z, freqs, qs
        
        #Get signal values froms states
        shape=(1,1,len(self.qxs)) #add z and fequency axis
        weights=self.wqxs.reshape(shape) #add z and frequency axes
        dipole_moments=self.get_dipole_moments(self.qxs).reshape(shape) #dipole moment for each q-value
        
        #self.dipole_moments=dipole_moments
        #self.ambient_rp=ambient_rp
        
        self.dipole_moments1=dipole_moments.copy()
        
        ##In case there's an ambient rp, update the radiated field##
        if ambient_rp is not None:
            ambient_rp.resize((1,len(freqs),1))
            dipole_moments_refl=self.get_dipole_moments_refl(self.qxs).reshape(shape)
            dipole_moments=dipole_moments+ambient_rp*dipole_moments_refl
            
        self.dipole_moments2=dipole_moments.copy()
        
        signals=numpy.sum(weights*psis*dipole_moments,axis=-1)
        signals2=weights*psis*dipole_moments
        
        #Demodulate if desired#
        d={}
        d['psi']=psis.squeeze()
        d['signals']=signals.squeeze()
        
        if demodulate:
            
            signal_vs_time,demodulated_signals=self.demodulate(signals,Nts=Nts,harmonics=harmonics)
            d['signal_vs_time']=signal_vs_time
            d.update(dict([(('signal_%i'%harmonic),demodulated_signals.cslice[harmonic]) \
                           for harmonic in harmonics]))
            
            if len(freqs)==1:
                psis=psis.squeeze()
                state_vs_time,demodulated_states=self.demodulate(psis,Nts=Nts,harmonics=harmonics)
                d['state_vs_time']=state_vs_time
                d.update(dict([(('state_%i'%harmonic),demodulated_states.cslice[harmonic]) \
                               for harmonic in harmonics]))
                signals2_vs_time,demodulated_signals2=self.demodulate(signals2.squeeze(),Nts=Nts,harmonics=harmonics)
                d.update(dict([(('signal2_%i'%harmonic),demodulated_signals2.cslice[harmonic]) \
                               for harmonic in harmonics]))
            
        return d
    
LightningRodModel=_LightningRodModel_()
LRM=LightningRodModel

class _LightningRodModel2_(_LightningRodModel_):
    """An experimental model where a different scattering matrix is loaded and used for each frequency."""
        
    
    def Lambda0Vector(self,qxs):
        
        Lambda0=self.Lambda0.interpolate_axis(qxs,axis=0,extrapolate=True)
        
        return numpy.matrix(Lambda0).T
    
    def Lambda0VectorRefl(self,qxs):
        
        Lambda0Refl=self.Lambda0Refl.interpolate_axis(qxs,axis=0,extrapolate=True)
        
        return numpy.matrix(Lambda0Refl).T
    
    def LambdaMatrix(self,qxs):
        
        #Should be no bounds error, we only interpolate into sampled region
        Lambda=self.Lambda.interpolate_axis(qxs,axis=0,kind='linear')\
                         .interpolate_axis(qxs,axis=1,kind='linear')
                         
        return numpy.matrix(Lambda)
        
    
    def evaluate_rp(self,freq,rp,qxs,**rpkwargs):
        
        a=self.geometric_params['a'] #used to convert qxs from units of 1/a
        k=2*numpy.pi*freq
        
        #exclude near the light line as comparatively unimportant
        if self.resonant_sample: Qs=numpy.sqrt(k**2+(qxs/(a*1e-7))**2)
        else: Qs=qxs/(a*1e-7)
        
        #Evaluate surface response#
        if hasattr(rp,'__call__'):
            #evaluate with q in units of cm-1 rather than 1/a
            rp=rp(freq, Qs, **rpkwargs)
        elif isinstance(rp,AWA) and 'Frequency' in rp.axis_names[0]:
            rp=rp.cslice[freq]
        
        self.rp=rp
        
        if isinstance(rp,numpy.ndarray):
            Logger.raiseException('If "rp" evaluates to an array, it must be of dimension 0, 1 or 2.',\
                                   unless=(rp.ndim in [0,1,2]), exception=ValueError)
            #G matrix evaluation will take care of turning rp into a matrix
        else:
            Logger.raiseException('"rp" must evaluate to a scalar or a 0-, 1- or 2-dimensional ndarray.',\
                                  unless=(type(rp) in number_types), exception=ValueError)
        
        return rp
        
    
    def GMatrix(self,z,freq,rp):
        
        a=self.geometric_params['a']
        knorm=2*numpy.pi*a*1e-7*freq
        
        if isinstance(rp,numpy.ndarray) and rp.ndim is 2:
            #Reflection matrix integral operator - it's not responsible for including its own weights (how would it know?)
            R=numpy.matrix(rp)
            #Propagator
            P=numpy.matrix(numpy.diag(numpy.exp(-self.qxs*z)))
            #Distribution of momenta
            D=numpy.matrix(numpy.diag(self.qxs))
            #If R is diagonal, then everything will be diagonal and will mutually commute
            W=numpy.matrix(numpy.diag(self.wqxs))
            
            return -D*P*W*R*P #note negative sign, to maintain typical notion of G in scattering problem
        
        else:
            pref=self.qxs
            #pref=(self.qxs**2+knorm**2)**(3/2.)/self.qxs**2 #; print 'Full calculation!' #Works for SiO2, less so for SiC
            return -numpy.matrix(numpy.diag(pref\
                                            *numpy.exp(-2*self.qxs*z)\
                                            *rp))
        
    def get_signal(self,freqs,rp,ambient_rp=None,zs=None,Nzs=20,zmin=1e-1,amplitude=80,\
                   a=None,interpolation=None,\
                   Nqs=72,demodulate=True,Nts=None,harmonics=[0,1,2,3],\
                   **kwargs):
        """qmin and qmax in units of 1/a"""
        
        if verbose:
            Logger.write('Computing near-field signal with Lightning Rod Model...')
        
        #Make sure frequencies are iterable
        if not hasattr(freqs,'__len__'): freqs=[freqs]
        freqs=numpy.array(freqs)
        
        #Prepare the model#
        self.prepare_model(zs=zs,zmin=zmin,Nzs=Nzs,Nqs=Nqs,amplitude=amplitude,a=a,kwargs=kwargs)
        
        ##In case there's an ambient rp##
        if ambient_rp is not None:
            ambient_rp=ambient_rp(freqs,angle=self.geometric_params['incidence_angle'])
        
        from numpy.lib.scimath import sqrt as cmplx_sqrt
        
        #Compute psi v. (freq, q) at each z
        psis=[]
        for i,freq in enumerate(freqs):
            progress=i/float(len(freqs))*100.
            if verbose: Logger.write('\tPROGRESS: %i%% - Computing state of the tip charge v. '%progress+\
                                     'z & s at freq=%s cm^-1...'%freq)
            
            a=self.geometric_params['a']
            knorm=2*numpy.pi*a*1e-7*freq
            self.kappas=cmplx_sqrt(self.qxs**2-knorm**2).real
            
            #Compute necessary matrices on our q-quadrature grid
            self.Lambda0Vecx=self.Lambda0Vector()
            self.LambdaMatx=self.LambdaMatrix()
            
            ##In case there's an ambient rp, update the initial charge distribution##
            if ambient_rp is not None:
                Lambda0Vecx=self.Lambda0Vecx+ambient_rp.cslice[freq]*self.Lambda0VectorRefl(self.qxs)
            else: Lambda0Vecx=self.Lambda0Vecx
            
            #Get distribution of charge w.r.t. original q-values
            rp_at_freq=self.evaluate_rp(freq,rp,self.qxs,**kwargs)
            psis_at_freq=[self.compute_state(Lambda0Vecx,self.LambdaMatx,\
                                            z,freq,rp=rp_at_freq) \
                                for z in self.zs[::-1]][::-1]
            psis.append(psis_at_freq)
            
        #Make into master array
        a=self.geometric_params['a']
        zs_nm=self.zs*a
        psis=Spectrum(psis,axes=[freqs,zs_nm,self.qxs],\
                      axis_names=['Frequency','Z [nm]','q [1/a]'],axis=None) #don't tell to FFT along any new axes!
        
        #Permute the axes to make zs first dimension (for convenience)
        psis=psis.transpose((1,0,2)) #z, freqs, qs
        
        #Get signal values froms states
        shape=(1,1,len(self.qxs)) #add z and fequency axis
        weights=self.wqxs.reshape(shape) #add z and frequency axes
        dipole_moments=self.get_dipole_moments(self.qxs).reshape(shape) #dipole moment for each q-value
        
        #self.dipole_moments=dipole_moments
        #self.ambient_rp=ambient_rp
        
        self.dipole_moments1=dipole_moments
        
        
        ##In case there's an ambient rp, update the radiated field##
        if ambient_rp is not None:
            ambient_rp.resize((1,len(freqs),1))
            dipole_moments_refl=self.get_dipole_moments_refl(self.qxs).reshape(shape)
            dipole_moments=dipole_moments+ambient_rp*dipole_moments_refl
            
        self.dipole_moments2=dipole_moments
        
        signals=numpy.sum(weights*psis*dipole_moments,axis=-1)
        
        #Demodulate if desired#
        d={}
        d['psi']=psis.squeeze()
        d['signals']=signals.squeeze()
        
        if demodulate:
            
            demodulated_signals=self.demodulate(signals,Nts=Nts,harmonics=harmonics)
            d.update(demodulated_signals)
            
        return d
    
LightningRodModel2=_LightningRodModel2_()
LRM2=LightningRodModel2



def ApproachCurve(freq,rp,zmin=.001,zmax=150,amplitude=40,a=10,Nts=50,\
                  Nzs=100,harmonics=[1,2,3],model=LRM,\
                  **kwargs):
    
    actual_zmax=zmax+amplitude
    actual_Nzs=int(Nzs*actual_zmax/float(zmax))
    dz=actual_zmax-zmin
    
    signal=model(freq,rp=rp,zmin=zmin,a=a,amplitude=dz,\
                 Nzs=actual_Nzs,normalize_to=None,demodulate=False,**kwargs)
    if isinstance(signal,dict): signal=signal['signals']
    
    ts=numpy.linspace(0,.5,Nts) #Need only integrate over half period
    
    approach_curves={}
    
    for harmonic in harmonics:
        
        z_axis=[]
        approach_curve=[]
        
        for zmin in signal.cslice[:zmax].axes[0]:
            
            #Interpolate at desired z-values#
            try: #might be a ValueError at interpolation limit
                new_zs=zmin+amplitude/2.*(1-numpy.cos(2*numpy.pi*ts))
                new_signal=signal.interpolate_axis(new_zs,axis=0,kind='slinear')
                
                #Demodulate
                ts_shape=(Nts,)+(1,)*(signal.ndim-1)
                ts_grid=ts.reshape(ts_shape)
                approach_value=simps(numpy.cos(2*numpy.pi*harmonic*ts_grid)*new_signal,\
                                     x=ts,axis=0) #Integrate over z(t) axis
                approach_curve.append(approach_value)
                z_axis.append(zmin)
                
            except ValueError: break
    
        #Turn result into ArrayWithAxes#
        axes=[z_axis]; axis_names=['Z']
        approach_curves[harmonic]=AWA(approach_curve,axes=axes,axis_names=axis_names)
    
    approach_curves['signal']=signal
    
    return approach_curves

def build_charge_distributions(geometry='cone',L=19000/30.,Rtop=0,Nzs=244,taper_angle=20,skin_depth=0.05,\
                               Nqs=244,qmin=1e-4,quadrature='TS',\
                               freq=30e-7*1000,find_q_cutoff=True,interpolate_zs=False,\
                               beam_shapes=['plane_wave','gaussian','linear'],\
                               incidence_angles=[30,45,60,90],\
                               **kwargs):
    
    quadrature_type=quadrature
    Logger.write('Using q-quadrature type: %s'%quadrature_type)
    Logger.write('\tUsing freq=%s...'%freq)
    if quadrature=='GL': quadrature=numrec.GL
    elif quadrature=='TS': quadrature=numrec.TS
    
    #Parameters relating to geometry and quadrature#
    if geometry=='sphere': L=2
    elif geometry=='ellipsoid' and L==2: geometry='sphere'
    qs,wqs=numrec.GetQuadrature(N=Nqs,xmin=qmin,xmax=numpy.inf,quadrature=quadrature) #units 1/a
    zs,wzs=numrec.GetQuadrature(N=Nzs,xmin=1e-3,xmax=L,quadrature=quadrature) #units of a (10 micron total length)
    zs-=zs.min() #Shift to lie on zero
    Rs=az.get_radii(zs,z0=0,L=L,R=1,taper_angle=taper_angle,geometry=geometry) #units of a
    
    #Parameters for integral transforms
    global pref_grid,exp_grid,zs_grid,ss_grid,Rs_grid,Qs_grid,wzs_grid,zs_interp,q,minz,maxz
    zs_grid=zs.reshape((len(zs),1))
    wzs_grid=wzs.reshape((len(wzs),1))
    Rs_grid=Rs.reshape((len(Rs),1))
    ss_grid=qs.reshape((1,len(qs)))
    k=2*numpy.pi*freq
    Qs_grid=numpy.sqrt(ss_grid**2+k**2)
    
    ##Integral transform kernel##
    if skin_depth:
        skin_depth=numpy.float(skin_depth)
        factor=1#-1j
        delta=skin_depth/factor
        exp_grid=(numpy.exp(-ss_grid*zs_grid)-numpy.exp(-zs_grid/delta))\
                  /(1-ss_grid*delta)
    else:
        exp_grid=numpy.exp(-ss_grid*zs_grid)
    
    pref_grid=exp_grid*j0(Qs_grid*Rs_grid) #zs and Rs units of a, ss units of 1/a
    where_faulty=(numpy.abs(pref_grid)>exp_grid)
    pref_grid[where_faulty]=0 #2020.03.25: what is this??

    #Parameters relating to dipole moment of a charge distribution
    for angle in incidence_angles:
        compliment_angle=180-angle
        if not compliment_angle in incidence_angles:
            incidence_angles.append(compliment_angle)
            
    RadPerChargeRing=RadPerChargeRingGenerator(zs,Rs,wzs=wzs,freq=freq)
    dipole_moments=dict((theta,[]) for theta in incidence_angles)
    charges=[]
    mean_errors=[]
    deviations=[]
    integral_xforms=[]
    
    Logger.write('\tBuilding charge distributions...')
    az.reuse_kernel=False
    for i,q in enumerate(qs):
        
        Logger.write('\tComputing charge for q*a=%1.2G -- Progress: %s%%'%(q,i/float(len(qs))*100))
        
        az.exp_decay=1/float(q) #in units of a
        
        charge,deviation,mean_error=az.get_charge_dist(z0=0,L=L,R=1,Rtop=Rtop,taper_angle=taper_angle,\
                                                       geometry=geometry,\
                                                       V_ext='exponential',\
                                                       quadrature=(zs,wzs),\
                                                       freq=freq,\
                                                       **kwargs)
        az.reuse_kernel=True #from now on, we can simply re-use computed kernel
        
        charges.append(charge)
        mean_errors.append(mean_error)
        deviations.append(deviation)
        Logger.write('\tMean error in boundary potential: %s%%'%(mean_error*100))
        
        Logger.write('\tComputing integral x-form...')
        charge_grid=charge.reshape((len(zs),1))
        integrand=pref_grid*charge_grid
        
        if interpolate_zs:
            minz=numpy.min((1e-3/q,.1*zs.min())); maxz=numpy.min((1e3/q,L))
            zs_interp=numpy.logspace(numpy.log10(minz),numpy.log10(maxz),10*Nzs)
            #zs_interp,_=numrec.GetQuadrature(N=8*Nzs,xmin=1e-3,xmax=L,quadrature=quadrature)
            x,y=integrand.axes
            spl1=RectBivariateSpline(x,y,integrand.real,s=0)
            spl2=RectBivariateSpline(x,y,integrand.imag,s=0)
            integrand_interp=spl1(zs_interp,y)+1j*spl2(zs_interp,y)
            integral_xform=simps(x=zs_interp,y=integrand_interp,axis=0)
        
        else:
            integral_xform=numpy.sum(wzs_grid*integrand,axis=0)
        
        #If a cutoff exists, it's where we find a negative peak s.t. q_peak>1
        #2020.03.25: what is this??
        if find_q_cutoff:
            neg_peaks=numrec.peakdetect(qs*numpy.abs(integral_xform),lookahead=10)[1]
            if len(neg_peaks):
                global inds,vals
                inds,vals=list(zip(*neg_peaks))
                for ind in inds:
                    if qs[ind]>1:
                        Logger.write('\tFound cutoff at q=%s'%qs[ind])
                        integral_xform[ind:]=0 #eliminate fictitious values beyond cutoff
                        break
                    
        integral_xforms.append(integral_xform)
        
        Logger.write('\tComputing dipole moment...')
        for theta in incidence_angles:
            dipole_moment_at_theta=RadPerChargeRing(charge,theta=theta)
            dipole_moments[theta].append(dipole_moment_at_theta)
        #dipole_moments.append(numpy.sum(wzs*zs*charge)) #Compares identically with above at freq=0
        
    charges=AWA(charges,axes=[qs,zs],axis_names=['q','z'])
    Rs=AWA(Rs,axes=[zs],axis_names=['z'])
    wzs=AWA(wzs,axes=[zs],axis_names=['z'])
    mean_errors=AWA(mean_errors,axes=[qs],axis_names=['q'])
    deviations=AWA(deviations,axes=[qs,zs],axis_names=['q','z'])
    integral_xforms=AWA(integral_xforms,axes=[qs,qs],axis_names=['q','s'])
    dipole_moments=dict([(theta,AWA(dipole_moments[theta],axes=[qs],axis_names=['q'])) \
                         for theta in incidence_angles])
    
    d={'charges':charges,\
       'integral_xforms':integral_xforms,\
       'charge_radii':Rs,\
       'charge_quadrature':(zs,wzs),\
        'mean_errors':mean_errors,\
        'deviations':deviations,\
        'dipole_moments':dipole_moments,\
        'quadrature':(qs,wqs)}
    
    for beam_shape in beam_shapes:
        for angle in incidence_angles:
            
            V_ext=beam_shape
            if beam_shape not in ['linear','point','exponential']: V_ext+=str(angle)
        
            Logger.write('Evaluating charge for "%s" beam profile...'%V_ext)
            charge,deviation,mean_error=az.get_charge_dist(z0=0,L=L,R=1,Rtop=Rtop,taper_angle=taper_angle,\
                                                               geometry=geometry,\
                                                               V_ext=V_ext,\
                                                               quadrature=(zs,wzs),\
                                                               freq=freq,\
                                                               **kwargs)
            d['charges_%s'%V_ext]=charge
            
            charge_grid=charge.reshape((len(zs),1))
            integrand=pref_grid*charge_grid
            
            if interpolate_zs:
                minz=numpy.min((1e-3/q,.1*zs.min())); maxz=numpy.min((1e3/q,L))
                zs_interp=numpy.logspace(numpy.log10(minz),numpy.log10(maxz),10*Nzs)
                #zs_interp,_=numrec.GetQuadrature(N=8*Nzs,xmin=1e-3,xmax=L,quadrature=quadrature)
                x,y=integrand.axes
                spl1=RectBivariateSpline(x,y,integrand.real)
                spl2=RectBivariateSpline(x,y,integrand.imag)
                integrand_interp=spl1(zs_interp,y)+1j*spl2(zs_interp,y)
                integral_xform=simps(x=zs_interp,y=integrand_interp,axis=0)
                
            else:
                integral_xform=numpy.sum(wzs_grid*integrand,axis=0)
            
            #If a cutoff exists, it's where we find a negative peak s.t. q_peak>1
            if find_q_cutoff:
                neg_peaks=numrec.peakdetect(qs*numpy.abs(integral_xform),lookahead=10)[1]
                if len(neg_peaks):
                    inds,vals=list(zip(*neg_peaks)); i=0; q_peak=qs[inds[i]]
                    while i<len(inds) and qs[inds[i]]<1: i+=1
                    if qs[inds[i]]>1:
                        print('Found cutoff at q=%s'%qs[inds[i]])
                        integral_xform[inds[i]:]=0 #eliminate fictitious values beyond cutoff
            
            integral_xform=AWA(integral_xform,axes=[qs],axis_names=['s'])
            
            d['integral_xforms_%s'%V_ext]=integral_xform
            
    filepath=get_charge_data_path(geometry,L,skin_depth,taper_angle,quadrature_type,Nzs,Nqs,freq)
    
    Logger.write('Writing charge data to file:\n\t"%s"'%filepath)
    file=open(filepath,'wb')
    pickle.dump(d,file); file.close()
    
    az.reuse_kernel=False
    
    return d

def RadPerChargeRingGenerator(zs,Rs,freq,wzs=None):
    
    dRs=differentiate(x=zs,y=Rs)
    nzs=1#/numpy.sqrt(1+dRs**2)
    nrs=dRs#/numpy.sqrt(1+dRs**2)
    
    global WLowerTri,zphases,rad
    N=len(zs)
    if wzs is None:
        wzs=numpy.diff(zs).tolist()
        wzs=numpy.array(wzs+[wzs[-1]])
    WLowerTri=numpy.matrix([wzs]*N) #weights run along columns
    WLowerTri[numpy.triu_indices(N,-1)]=0 #make upper triangle zero above diagonal
    numpy.fill_diagonal(WLowerTri,.5*wzs) #half-weights on diagonal
    
    def RadPerChargeRing(charge_dist,theta=90):
        
        global zphase,rad
        theta=theta*numpy.pi/180.
        if hasattr(theta,'__len__'):
            theta=numpy.array(theta).reshape((len(theta),1))
        
        #first -1 is just a prefactor before the green's function
        Rnorms=2*numpy.pi*freq*Rs
        znorms=2*numpy.pi*freq*zs
        zcurrentlobe=-1*numpy.abs(numpy.sin(theta))*j0(Rnorms*numpy.abs(numpy.sin(theta))) #For positive (upwards) z component
        rcurrentlobe=-1*1j*numpy.cos(theta)*j1(Rnorms*numpy.abs(numpy.sin(theta)))
        zphases=numpy.exp(-1j*znorms*numpy.cos(theta))
        
        #sin(theta) for solid angle weight
        E_per_current_ring=numpy.matrix((nzs*zcurrentlobe+nrs*rcurrentlobe)*zphases).T
        E_per_charge_ring=numpy.array(WLowerTri*E_per_current_ring).squeeze()
        
        if hasattr(theta,'__len__'): integrand=(wzs*charge_dist).reshape((len(wzs),1))\
                                                *E_per_charge_ring
        else: integrand=wzs*E_per_charge_ring*charge_dist
            
        rad=numpy.sum(integrand,axis=0)        
        if isinstance(rad,numpy.ndarray):
            if rad.ndim: rad=AWA(rad,axes=[theta.flatten()],\
                                axis_names=['Angle (radians)'])
            else: rad=rad.tolist()
        
        return rad
    
    return RadPerChargeRing