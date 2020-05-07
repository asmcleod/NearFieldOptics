#Materials definitions for Graphene
"""All frequency/wave vector quantities are in units of 1/cm"""

import os
import sys
import numpy; pi=numpy.pi
import scipy
import cmath
import copy
import pickle
import numbers
from common.log import Logger
from common import misc
from common.baseclasses import ArrayWithAxes as AWA
from common import numerics
from scipy.special import erf
from numpy import linalg
from NearFieldOptics.Materials import anisotropic
from NearFieldOptics.Materials import faddeeva

def erfc(x): return 1-erf(x)

def ensure_complex(val):

    if isinstance(val,numpy.ndarray):
        try: return val.astype('complex')
        #Sometimes *val* is an array with *object* type elements (unconvertible)
        except TypeError: return val
    else: return numpy.complex(val)

def safe_sqrt(val,flip_lt=0):
    #Make sure sqrt results in solution in the first quadrant of the complex plane
    
    val=ensure_complex(numpy.sqrt(val))
    
    ##Enforce positive imaginary part - decaying in physics convention##
    val=numpy.where((numpy.imag(val)<flip_lt),\
                    -val, val)
    
    return val

def _vectorize_cmath_():

    ####Some names made to be usable variables####
    cmath_dict=cmath.__dict__
    for key in list(cmath_dict.keys()):
        func=cmath_dict[key]
        if not hasattr(func,'__call__'): continue
        elif key=='sqrt': cmath_dict[key]=safe_sqrt
        else: cmath_dict[key]=numpy.frompyfunc(func,1,1)
        
def _prepare_freq_and_q_holder_(freq,q,\
                                angle=None,\
                                entrance=None):
    
    #Convert angle to q
    if angle!=None:
        assert isinstance(angle,numbers.Number),'`angle` must be a single number.'
        if not entrance: entrance=Air
        angle_rad=angle/180.*pi
        k=safe_sqrt(entrance.optical_constants(freq))*freq
        q=numpy.real(k*numpy.sin(angle_rad))
        
    ##Prepare AWA if there are axes in *freq* and *q*##
    freq,q=numerics.broadcast_items(freq,q)
        
    axes=[]; axis_names=[]
    if isinstance(freq,numpy.ndarray):
        freqaxis=freq.squeeze()
        if not freqaxis.ndim: freqaxis.resize((1,))
        axes.append(freqaxis); axis_names.append('Frequency')
    if isinstance(q,numpy.ndarray) and angle is None:
        qaxis=q.squeeze()
        if not qaxis.ndim: qaxis.resize((1,))
        axes.append(qaxis); axis_names.append('q-vector')
    if axes:
        shape=[len(axis) for axis in axes]
        holder=AWA(numpy.zeros(shape),axes=axes,axis_names=axis_names)
    else: holder=0
    
    return freq,q,ensure_complex(holder)

_vectorize_cmath_()

c=3e10 #units of cm/s

class Material(object):
    
    def __init__(self,eps_infinity=1,mu_infinity=1):
        
        self.eps_infinity=eps_infinity
        self.mu_infinity=mu_infinity

    def epsilon(self,freq,q=None): return self.eps_infinity
    
    def mu(self,freq,q=None): return self.mu_infinity

    def optical_constants(self,freq,q=0):
        
        eps=self.epsilon(freq,q)
        mu=self.mu(freq,q)
        
        result=safe_sqrt(eps*mu)
        
        return ensure_complex(result)
    
    def get_kz(self,freq,q=0,eps=None,mu=None,QS=False): 
        
        if QS: return 1j*q
        
        if eps is not None and mu is not None:
            n=safe_sqrt(eps*mu)
        else: n=self.optical_constants(freq,q)
        
        omega=2*pi*freq
        kz=safe_sqrt((n*omega)**2-q**2)
        
        return ensure_complex(kz)
    
    def get_p_Ez(self,freq,q,Ex=1,eps=None,mu=None,kz=None):
        
        if kz is None: kz=self.get_kz(freq,q,eps=eps,mu=mu)
        
        return -q/kz*Ex

class BaseIsotropicMaterial(Material):

    def reflection_p(self,freq,q=0,angle=None,\
                     entrance=None,entrance_kz=None,\
                     surface=None,screening=1,**kwargs):
        
        if entrance is None:
            assert entrance_kz is None,'`entrance_kz` can only be specified when `entrance` material is specified as well!'
            entrance=Air
        if entrance_kz is None: entrance_kz=entrance.get_kz
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        kz1=entrance_kz(freq,q)
        kz2=self.get_kz(freq,q,**kwargs)
        
        eps1=entrance.epsilon(freq,q)
        eps2=self.epsilon(freq,q)*screening
        
        omega=2*pi*freq
        
        if surface:
            if hasattr(surface,'__len__'):
                sigma=numpy.sum([surface.conductivity(freq) for \
                                 surface in surface], axis=0)
            else: sigma=surface.conductivity(freq)
            surf=4*pi*kz1*kz2*sigma/(c*omega)
        else: surf=0
        
        rp+=(eps2*kz1-eps1*kz2+surf)/\
            (eps2*kz1+eps1*kz2+surf)
            
        self.kz1=kz1
        self.kz2=kz2
        self.eps1=eps1
        self.eps2=eps2
               
        return ensure_complex(rp)
    
    def transmission_p(self,freq,q=0,angle=None,\
                     entrance=None,entrance_kz=None,\
                     surface=None,screening=1,\
                     magnetic=False,**kwargs):
        
        if entrance is None:
            assert entrance_kz is None,'`entrance_kz` can only be specified when `entrance` material is specified as well!'
            entrance=Air
        if entrance_kz is None: entrance_kz=entrance.get_kz
        
        ##Get holder for data, and expanded freq & q##
        freq,q,tp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)

        kz1=entrance_kz(freq,q)
        kz2=self.get_kz(freq,q)
        
        eps1=entrance.epsilon(freq,q)
        eps2=self.epsilon(freq,q)*screening
        
        omega=2*pi*freq
        
        #@TODO: implement surface here
        surf=0
        
        tp+=2*eps2*kz1/\
            (eps2*kz1+eps1*kz2+surf)
        if not magnetic:
            tp*=safe_sqrt(eps1/eps2)
            
        return ensure_complex(tp)

    def reflection_s(self,freq,q=0,angle=None,\
                     entrance=None,entrance_kz=None,\
                     surface=None,**kwargs):
        
        if entrance is None:
            assert entrance_kz is None,'`entrance_kz` can only be specified when `entrance` material is specified as well!'
            entrance=Air
        if entrance_kz is None: entrance_kz=entrance.get_kz
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        kz1=entrance_kz(freq,q)
        kz2=self.get_kz(freq,q)
        
        omega=2*pi*freq
        
        if surface:
            if hasattr(surface,'__len__'):
                sigma=numpy.sum([surface.conductivity(freq) for \
                                 surface in surface], axis=0)
            else: sigma=surface.conductivity(freq)
            surf=4*pi*kz1*kz2*sigma/(c*omega)
        else: surf=0
        
        rp+=(kz1-kz2+surf)/\
            (kz1+kz2+surf)
               
        return ensure_complex(rp)

Air=BaseIsotropicMaterial(eps_infinity=1)
Air.name='Air'

class DataMaterial(BaseIsotropicMaterial):
    
    def __init__(self,freqs,eps_values,mu_infinity=1):
        
        self.eps_values=AWA(eps_values,axes=[freqs],axis_names=['Frequency'])
        
        BaseIsotropicMaterial.__init__(self,eps_infinity=1,mu_infinity=mu_infinity)
        
    def epsilon(self,freq,q=None):
        
        return self.eps_values.interpolate_axis(freq,axis=0)

class IsotropicMaterial(BaseIsotropicMaterial):
    
    def __init__(self,eps_infinity=1,mu_infinity=1,\
                 drude_params=[],\
                 phonon_params=[],\
                 eps_lps=[],mu_lps=[],\
                 eps_vps=[],mu_vps=[]):
        
        self.drude_params=list(drude_params)
        self.phonon_params=list(phonon_params)
        
        self.eps_lps=list(eps_lps)
        self.mu_lps=list(mu_lps)
        
        self.eps_vps=list(eps_vps)
        self.mu_vps=list(mu_vps)
        
        BaseIsotropicMaterial.__init__(self,eps_infinity,mu_infinity)
        
    def get_drude(self,freq,q,drude_params=[]):
        
        if len(drude_params) and not hasattr(drude_params[0],'__len__'): drude_params=[drude_params]
        
        result=0
        for D in drude_params:
            
             beta=1
             plasma_f,damping_f=D
             sigma=plasma_f**2/(4*numpy.pi*damping_f)/(1-1j*freq/damping_f)**beta
             result+=4*numpy.pi*1j*sigma/freq
            
             #plasma_f,damping_f=D
             #result+=-plasma_f**2/(freq*(freq+1j*damping_f))
        
        return result

    def get_phonons(self,freq,q,eps_infinity=1,phonon_params=[]):
        
        if len(phonon_params) and not hasattr(phonon_params[0],'__len__'): phonon_params=[phonon_params]
        
        #Assume phonon_params is of form [(w_LO1,g_LO1,w_TO1,g_TO1),...,(w_LOn,g_LOn,w_TOn,g_TOn)]
        result=eps_infinity
        for phonon_mode in phonon_params:
            w_LO,g_LO,w_TO,g_TO=phonon_mode
            result*=(w_LO**2-freq**2-1j*freq*g_LO)/\
                    (w_TO**2-freq**2-1j*freq*g_TO)
                    
        #subtract eps_infinity because it's "built-in" both here and in the definition of epsilon
        return result-eps_infinity 
        
    def get_lorentzians(self,freq,q,lps=[]):
        
        if len(lps) and not hasattr(lps[0],'__len__'): lps=[lps]
        
        result=0
        for L in lps:
            amp,resonance,damping=L
            result+=amp/(resonance**2-freq**2-1j*damping*freq)
        
        return result
    
    ##These are unused right now, but could be used in a rational approximant to the Voigt function##
    approx_voigt_consts={'A':[-1.2150,  -1.3509,    -1.2150,    -1.3509],\
                         'B':[1.2359,    0.3786,    -1.2359,    -0.3786],\
                         'C':[-0.3085,   0.5906,    -0.3085,     0.5906],\
                         'D':[0.0210,   -1.1858,    -0.0210,     1.1858]}
        
    #Formula used by Meneses et. al., 2004
    def get_Meneses_peak(self,freq,q,eps_vps=[]):
        
        result=0
        for V in self.eps_vps:
            A,w,gamma_L,gamma_G=V
            
            global x, x0, y
            x=2*numpy.sqrt(numpy.log(2))/float(gamma_G)*freq
            x0=2*numpy.sqrt(numpy.log(2))/float(gamma_G)*w
            y=gamma_L/float(gamma_G)*numpy.sqrt(numpy.log(2))
            
            wminus=faddeeva.faddeeva(x-x0+1j*y)
            #if numpy.isnan(wminus).any(): return x-x0+1j*y,wminus
            wplus=faddeeva.faddeeva(x+x0+1j*y)
            denom=numpy.real(faddeeva.faddeeva(1j*y))
            
            result+=A/denom*(-numpy.imag(wminus+wplus)+1j*numpy.real(wminus-wplus))
        
        return result
    
    ##Formula used by Kucirkova & Navratil, 1994
    def get_voigt_profile(self,freq,q,eps_vps=[]):
        
        result=0
        for V in eps_vps:
            A,w,gamma_L,gamma_G=V
            
            wminus=faddeeva.faddeeva((freq-w)/float(gamma_G)+1j*gamma_L/float(2*gamma_G))
            wplus=faddeeva.faddeeva((freq+w)/float(gamma_G)+1j*gamma_L/float(2*gamma_G))
            result+=A*1j*(wminus-wplus)

        return result
    
    eps_property_functions={'get_drude':'drude_params',\
                            'get_phonons':('eps_infinity','phonon_params'),\
                            'get_lorentzians':'eps_lps',\
                            'get_voigt_profile':'eps_vps'}
    mu_property_functions={'get_lorentzians':'mu_lps',\
                           'get_voigt_profile':'mu_vps'}
    
    def invoke_property_function(self,freq,q,prop_func,props):
        
        if not props: return 0
        if isinstance(prop_func,str): prop_func=getattr(self,prop_func)
        
        #Properties should be made lists of property names
        if isinstance(props,str) or not hasattr(props,'__len__'): props=[props]
        else: props=list(props)
        
        #Request the corresponding attribute for each property in the prop list#
        for i,property in enumerate(props):
            if isinstance(property,str): props[i]=getattr(self,property)
        
        return prop_func(freq,q,*props)
    
    def epsilon(self,freq,q=None):
        
        eps=self.eps_infinity
        for prop_func,props in self.eps_property_functions.items():
            eps+=self.invoke_property_function(freq, q, prop_func, props)
                
        return ensure_complex(eps)
    
    def mu(self,freq,q=None):
        
        mu=self.mu_infinity
        for prop_func,props in self.mu_property_functions.items():
            mu+=self.invoke_property_function(freq, q, prop_func, props)
                
        return ensure_complex(mu)

class Semiconductor(IsotropicMaterial):
    
    def __init__(self,dopings,mobilities=50,meffs=1,eps_infinity=1,mu_infinity=1):
        """Carrier density `n` in cm^-3
        Mobility `mobility` in cm^2/(V*s)"""
        
        if not hasattr(dopings,'__len__'): dopings=[dopings]
        if not hasattr(mobilities,'__len__'): mobilities=[mobilities]
        if not hasattr(meffs,'__len__'): meffs=[meffs]
        
        self.dopings=dopings
        self.mobilities=mobilities
        self.meffs=meffs
        
        e_C=1.602176565e-19 #Coulombs
        m_kg=9.10938291e-31 #kg
        eps0=8.854187817620e-12 #SI units
        c_cm=3e10
        
        drude_params=[]
        for n,mobility,meff in zip(dopings,mobilities,meffs):
        
            n_m3=n*1e6 #per cubic meter
            mobility_m2=mobility*1e-4 #into m^2 rather than cm^2
            
            wp_rad=numpy.sqrt(n_m3*e_C**2/(m_kg*meff*eps0))
            wp_cm=wp_rad/(2*pi*c_cm)
            
            gamma_rad=(e_C)/(mobility_m2*m_kg*meff)
            gamma_cm=gamma_rad/(2*pi*c_cm)
            
            drude_params.append([wp_cm,gamma_cm])
        
        IsotropicMaterial.__init__(self,eps_infinity=eps_infinity,\
                                   mu_infinity=mu_infinity,\
                                   drude_params=drude_params)
        
    def set_doping(self,dopings):
        
        Semiconductor.__init__(self,dopings,mobilities=self.mobilities,\
                               meffs=self.meffs,eps_infinity=self.eps_infinity,\
                               mu_infinity=self.mu_infinity)
        
    def set_mobility(self,mobilities):
        
        Semiconductor.__init__(self,dopings=self.dopings,mobilities=mobilities,\
                               meffs=self.meffs,eps_infinity=self.eps_infinity,\
                               mu_infinity=self.mu_infinity)

class DopedSilicon(Semiconductor):
    
    data_dir=os.path.join(os.path.dirname(__file__),\
                          'Tabulated')
    
    def __init__(self,ne,nh=0):
        
        e_mobilities_file=os.path.join(self.data_dir,\
                                       'Silicon_ElectronMobilityVsDoping.csv')
        e_dopings,e_mobilities=misc.extract_array(open(e_mobilities_file)).astype(float).T
        e_dopings=10**e_dopings
        self.e_mobilities=AWA(e_mobilities,axes=[e_dopings],axis_names=['Electron doping'])
        
        h_mobilities_file=os.path.join(self.data_dir,\
                                       'Silicon_HoleMobilityVsDoping.csv')
        h_dopings,h_mobilities=misc.extract_array(open(h_mobilities_file)).astype(float).T
        h_dopings=10**h_dopings
        self.h_mobilities=AWA(h_mobilities,axes=[h_dopings],axis_names=['Hole doping'])
        
        self.set_doping(ne,nh)
        
    def set_doping(self,ne,nh=0):
        
        mobility_functions={'n':self.get_electron_mobility,\
                            'p':self.get_hole_mobility}
        effective_masses={'n':.26,\
                          'p':.37} #validated from several sources
        
        Semiconductor.__init__(self,dopings=[ne,nh],\
                               mobilities=[mobility_functions['n'](ne),\
                                           mobility_functions['p'](nh)],\
                               meffs=[effective_masses['n'],\
                                      effective_masses['p']],\
                               eps_infinity=11.9+1.3704e-6j,\
                               mu_infinity=1)
        
    def get_electron_mobility(self,ne):
        
        #return 65+1265/(1+(ne/8.5e16)**.72)
        return self.e_mobilities.interpolate_axis(ne,axis=0,bounds_error=False,extrapolate=True)*.75
    
    def get_hole_mobility(self,nh):
        
        #return 48+447/(1+(nh/6.3e16)**.76)
        return self.h_mobilities.interpolate_axis(nh,axis=0,bounds_error=False,extrapolate=True)

class TabulatedMaterial(BaseIsotropicMaterial):
    
    def __init__(self,eps_data,factor=1):
        
        self._eps_data=eps_data
        self.factor=factor
        
        BaseIsotropicMaterial.__init__(self)
        
    def epsilon(self,freq,q=0,**kwargs):
        
        return self.factor*\
                self._eps_data.interpolate_axis(freq,axis=0,**kwargs)
    
class TabulatedMagneticMaterial(BaseIsotropicMaterial):
    
    def __init__(self,eps_data,mu_data,factor=1):
        
        self._eps_data=eps_data
        self._mu_data=mu_data
        self.factor=factor
        
        BaseIsotropicMaterial.__init__(self)
        
    def mu(self,freq,q=0,**kwargs):
        
        return self.factor*\
                self._mu_data.interpolate_axis(freq,axis=0,**kwargs)
    
    
class TabulatedMaterialFromFile(TabulatedMaterial):
    
    data_dir=os.path.join(os.path.dirname(__file__),\
                          'Tabulated')
    
    def __init__(self,epsfile,factor=1):
        
        Logger.write('Loading tabulated material data from file "%s"...'%epsfile)
        if not os.path.exists(epsfile):
            epsfile=os.path.join(self.data_dir,epsfile)
        assert os.path.exists(epsfile),FileNotFoundError('file "%s" not found'%epsfile)
        file=open(epsfile)
        
        if epsfile.lower().endswith('.pickle'):
            file=open(epsfile,'rb')
            eps_data=pickle.load(file,encoding='bytes')
        elif epsfile.lower().endswith('.csv') or epsfile.lower().endswith('.txt'):
            freq,eps1,eps2=misc.extract_array(file, dtype=numpy.float).T
            eps_data=AWA(eps1+1j*eps2,axes=[freq],axis_names=['Frequency [cm^-1]'])
        else: Logger.error('File type not understood for file "%s".'%epsfile)
        
        file.close()
        
        TabulatedMaterial.__init__(self,eps_data,factor=factor)
        
class TabulatedMagneticMaterialFromFile(TabulatedMagneticMaterial):
    
    data_dir = os.path.join(os.path.dirname(__file__),\
                            'Tabulated')
    
    def __init__(self,epsmufile):
        
        Logger.write('Loading tabulated magnetic material data form file "%s"...'%epsmufile)
        if not os.path.exists(epsmufile):
            epsmufile=os.path.join(self.data_dir,epsmufile)
        assert os.path.exists(epsmufile),FileNotFoundError('file "%s" not found'%epsmufile)
        file=open(epsmufile)
        
        if epsmufile.lower().endswith('.pickle'):
            file=open(epsmufile,'rb')
            eps_data=pickle.load(file,encoding='bytes')
        elif epsmufile.lower().endswith('.csv') or epsmufile.lower().endswith('.txt'):
            freq,eps1,eps2,mu1,mu2=misc.extract_array(file, dtype=numpy.float).T
            eps_data=AWA(eps1+1j*eps2,axes=[freq],axis_names=['Frequency [cm^-1]'])
            mu_data = AWA(mu1+1j*mu2,axes = [freq],axis_names = ['Frequency [cm^-1]'])
        else: Logger.error('File type not understood for file "%s".'%epsmufile)
        
        file.close()
        
        TabulatedMagneticMaterial.__init__(self,eps_data,mu_data)
        
class Surface(object):
    
    def __init__(self,sigma0=0):
        
        self.sigma0=sigma0
        
    def conductivity(self,freq,enhancement=1): return self.sigma0*enhancement

class IsotropicSurface(Surface):
    
    def reflection_p(self,freq,q,angle=None,\
                     entrance=Air,exit=Air,screening=1,**kwargs):
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)

        eps1=entrance.epsilon(freq)
        eps2=exit.epsilon(freq)*screening

        kz1=entrance.get_kz(freq,q)
        kz2=exit.get_kz(freq,q)

        sigma=self.conductivity(freq,**kwargs)
        omega=2*pi*freq
        
        rp+=(eps2*kz1-eps1*kz2+4*pi*kz1*kz2*sigma/(c*omega))/\
           (eps2*kz1+eps1*kz2+4*pi*kz1*kz2*sigma/(c*omega))

        return ensure_complex(rp)
    
class TabulatedSurface(IsotropicSurface):
    
    def __init__(self,conductivity_data,factor=1):
        
        self._conductivity_data=conductivity_data
        self.factor=factor
        
        IsotropicSurface.__init__(self)
        
    def conductivity(self,freq,q=0):
        
        return self.factor*\
                self._conductivity_data.interpolate_axis(freq,axis=0)
    
class TabulatedSurfaceFromFile(TabulatedSurface):
    
    data_dir=os.path.join(os.path.dirname(__file__),\
                          'Tabulated')
    
    def __init__(self,conductivity_file,factor=1):
        
        Logger.write('Loading tabulated surface conductivity data from file "%s"...'%conductivity_file)
        if not os.path.exists(conductivity_file):
            conductivity_file=os.path.join(self.data_dir,conductivity_file)
        file=open(conductivity_file)
        
        if conductivity_file.lower().endswith('.pickle'):
            file=open(epsfile,'rb')
            conductivity_data=pickle.load(file,encoding='bytes')
        elif conductivity_file.lower().endswith('.csv'):
            freq,sigma1,sigma2=misc.extract_array(file, dtype=numpy.float).T
            conductivity_data=AWA(sigma1+1j*sigma2,axes=[freq],axis_names=['Frequency [cm^-1]'])
        else: Logger.error('File type not understood for file "%s".'%conductivity_file)
        
        file.close()
        
        TabulatedSurface.__init__(self,conductivity_data,factor=factor)

class SingleLayerGraphene(IsotropicSurface):

    def __init__(self,chemical_potential=1000,gamma=.1,layers=1):
        
        IsotropicSurface.__init__(self)
        
        self.chemical_potential=chemical_potential
        self.gamma=gamma
        self.layers=layers
        self.residual_conductivity=0

    def zeta_bar(self,freq):

        result=(freq+1j*self.gamma)/float(self.chemical_potential)
        
        return ensure_complex(result)
    
    def conductivity(self,freq,q=None):
        "Units of cm/sec for graphene"
        
        zeta_bar=self.zeta_bar(freq)
        f=(1-1/4.*zeta_bar*cmath.log((2.+zeta_bar)/(2.-zeta_bar)))
        
        alpha=1/137.
        sigma=1j*(c*alpha/pi)*f/zeta_bar*self.layers
        
        sigma+=(c*alpha/pi)*self.residual_conductivity

        return ensure_complex(sigma)

class TopologicalInsulatorSurface(SingleLayerGraphene):

    def __init__(self,chemical_potential=1000,gamma=30):
        
        #Should be identical to graphene in terms of functional form of conductivity (approximately in IR)
        SingleLayerGraphene.__init__(self, chemical_potential, gamma)
        
    def conductivity(self,freqs):
        
        #Return half graphene conductivity - one dirac cone per unit cell in density of states - graphene has 2
        return 1/4.*SingleLayerGraphene.conductivity(self,freqs)



class Layer(object):
    
    def __init__(self,material,thickness):
        
        self.set_material(material)
        self.set_thickness(thickness)
        
    def set_material(self,material):
        
        Logger.raiseException('*material* must be a *Material* instance.',\
                              exception=TypeError, unless=isinstance(material,Material))
        self._material=material
    
    def set_thickness(self,thickness):
        
        Logger.raiseException('*thickness* must be a positive numerical thickness '+\
                              'in units of cm.', exception=TypeError, \
                              unless=(type(thickness) in numerics.number_types \
                                      and thickness>0))
        self._thickness=thickness
    
    def get_material(self): return self._material
    
    def get_thickness(self): return self._thickness

class LayeredMedia(object):
    
    def __init__(self,*layers,**kwargs):
        
        self.set_layers(*layers)
        
        #Set default entrance/exit materials
        exkwargs=misc.extract_kwargs(kwargs,entrance=Air,exit=Air)
        self.set_entrance(exkwargs['entrance'])
        self.set_exit(exkwargs['exit'])
        
    def set_layers(self,*layers):
        
        Logger.raiseException('Please provide one or materials from which to construct '+\
                              'material layers.', exception=IndexError, unless=len(layers))
        
        verified_layers=[]
        for layer in layers:
            Logger.raiseException('Each provided material must either be of type *Surface* '+\
                                  'or be a (*Material*, *thickness*) tuple - with thickness in cm. '+\
                                  'Alternatively, another *LayeredMedia* can also be incorporated.',\
                                  exception=TypeError, unless=(isinstance(layer,(Surface,Layer,LayeredMedia)) \
                                                               or (hasattr(layer,'__len__') \
                                                               and len(layer)==2)) )
                
            if isinstance(layer,Surface):
                surface=layer
                verified_layers.append(surface)
                
            elif isinstance(layer,LayeredMedia):
                layered_media=layer
                verified_layers+=layered_media.get_layers()
                
            elif isinstance(layer,Layer):
                verified_layers.append(layer)
                
            elif hasattr(layer,'__len__'):
                material,thickness=layer
                verified_layers.append(Layer(material=material,\
                                             thickness=thickness))

        self._layers=verified_layers
        
    def set_entrance(self,entrance):
        
        Logger.raiseException('*entrance* must be a *Material* instance.',\
                              exception=TypeError,
                              unless=isinstance(entrance,Material))
        self._entrance=entrance
        
    def get_entrance(self): return self._entrance
    
    def get_exit(self): return self._exit
    
    def set_exit(self,exit):
        
        Logger.raiseException('*exit* must be a *Material* instance.',\
                              exception=TypeError,
                              unless=isinstance(exit,Material))
        self._exit=exit
        
    def get_layers(self): return copy.copy(self._layers)
    
    def reverse(self):
        """Reverse the order of layers in this *LayeredMedia*."""
        
        self._materials.reverse()
        
    def reflection_s(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Use Rouard's method to compute the composite Fresnel reflection
        coefficient of this *LayeredMedia* for p-polarized light incident
        from above through *entrance* with *exit* below the final layer."""
        self.rss=[]
        #Get entrance/exit materials
        if not entrance: entrance=self.get_entrance()
        else: entrance=Air
        if not exit: exit=self.get_exit()
        else: exit=Air
        layers=self.get_layers()
        
        ##Get holder for data, and expanded freq & q##
        #In this function, we only use these in this scope
        #for exponential propogator terms
        freq_grid,q_grid,rs=_prepare_freq_and_q_holder_(freq,q,\
                                                        angle=angle,\
                                                        entrance=entrance)
        
        ##Get rho at exit##
        below_material=exit
        #Get any surface materials at bottom#
        surfaces=[]
        while isinstance(layers[-1],Surface): surfaces.append(layers.pop())
        above_material=layers[-1].get_material()
        #Get rs for this interface#
        rs=below_material.reflection_s(freq,q,angle,\
                                        surface=surfaces,\
                                        entrance=above_material)
        rho=rs; self.rss.append(rs)
        
        #See how many bulk media we have#
        #equals the number of remaining interfaces to iterate through#
        number_bulk_media=[isinstance(layer,Layer) for layer in layers].count(True)
        
        #Make entrance a dummy "layer" for convenience
        layers=[Layer(entrance,thickness=1)]+layers
        
        ##Iterate over layers from bottom to build up rho##
        for i in range(number_bulk_media):
            
            below_layer=layers.pop()
            below_material=below_layer.get_material()
            
            #Again collect surface materials at next interface#
            surfaces=[]
            while isinstance(layers[-1],Surface): surfaces.append(layers.pop())
            
            #Get rs for this interface#
            above_material=layers[-1].get_material()
            if isinstance(above_material,BaseAnisotropicMaterial):
                entrance_kz=above_material.get_extraordinary_kz
            else: entrance_kz=above_material.get_kz
            rs=below_material.reflection_s(freq,q,angle,\
                                            surface=surfaces,\
                                            entrance_kz=entrance_kz)
            self.rss.append(rs)
            
            #Make rho#
            if isinstance(below_material,BaseAnisotropicMaterial):
                below_kz=below_material.get_extraordinary_kz
            else: below_kz=below_material.get_kz
            
            below_kz=below_kz(freq_grid,q_grid)
            below_thickness=below_layer.get_thickness()
            rho=(rs+rho*numpy.exp(-2*1j*below_kz*below_thickness))/\
                (1+rs*rho*numpy.exp(-2*1j*below_kz*below_thickness))
                
        return rho
        
    def reflection_p(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Use Rouard's method to compute the composite Fresnel reflection
        coefficient of this *LayeredMedia* for p-polarized light incident
        from above through *entrance* with *exit* below the final layer."""
        
        self.rps=[]
        #Get entrance/exit materials
        if entrance is None: entrance=self.get_entrance()
        else: entrance=Air
        if exit is None: exit=self.get_exit()
        else: exit=Air
        layers=self.get_layers()
        
        #Make entrance a dummy "layer" for convenience
        layers=[Layer(entrance,thickness=1)]+layers
        
        ##Get holder for data, and expanded freq & q##
        #In this function, we only use these in this scope
        #for exponential propogator terms
        freq_grid,q_grid,rp=_prepare_freq_and_q_holder_(freq,q,\
                                                        angle=angle,\
                                                        entrance=entrance)
        
        ##Get rho at exit##
        below_material=exit
        #Get any surface materials at bottom#
        surfaces=[]
        while isinstance(layers[-1],Surface):
            surfaces.append(layers.pop())
            if not len(layers): break
        above_material=layers[-1].get_material()
        #Get rp for this interface#
        if isinstance(above_material,BaseAnisotropicMaterial):
            entrance_kz=above_material.get_extraordinary_kz
        else: entrance_kz=above_material.get_kz
        rp=below_material.reflection_p(freq,q,angle,\
                                        surface=surfaces,\
                                        entrance=above_material,
                                        entrance_kz=entrance_kz,**kwargs)
        rho=rp; self.rps.append(rp)
        
        #See how many bulk media we have#
        #equals the number of remaining interfaces to iterate through#
        number_bulk_media=[isinstance(layer,Layer) for layer in layers].count(True)-1 #discount entrance layer
        
        ##Iterate over layers from bottom to build up rho##
        for i in range(number_bulk_media):
            
            below_layer=layers.pop(-1)
            below_material=below_layer.get_material()
            #print 'Material below: %s'%below_material.name
            
            #Again collect surface materials at next interface#
            surfaces=[]
            while isinstance(layers[-1],Surface): surfaces.append(layers.pop())
            
            #Get rp for this interface#
            above_material=layers[-1].get_material()
            if isinstance(above_material,BaseAnisotropicMaterial):
                entrance_kz=above_material.get_extraordinary_kz
            else: entrance_kz=above_material.get_kz
            rp=below_material.reflection_p(freq,q,angle,\
                                            surface=surfaces,\
                                            entrance=above_material,\
                                            entrance_kz=entrance_kz)
            self.rps.append(rp)
            
            #Make rho#
            if isinstance(below_material,BaseAnisotropicMaterial):
                below_kz=below_material.get_extraordinary_kz
            else: below_kz=below_material.get_kz
            
            below_kz=below_kz(freq_grid,q_grid)
            below_thickness=below_layer.get_thickness()
            rho=(rp+rho*numpy.exp(+2*1j*below_kz*below_thickness))/\
                (1+rp*rho*numpy.exp(+2*1j*below_kz*below_thickness))
                
        return rho
    
class BaseAnisotropicMaterial(Material):
    
    def __init__(self,eps_infinity=1,mu_infinity=1):
        
        if isinstance(eps_infinity,numerics.number_types): eps_infinity=[eps_infinity]*3
        if isinstance(mu_infinity,numerics.number_types): mu_infinity=[mu_infinity]*3
        Logger.raiseException('*eps_infinity* and *mu_infinity* should be either numbers, '+\
                              'or iterables of length three (one constant for each of the '+\
                              'principle axes, respectively).',\
                              unless=((isinstance(eps_infinity,numerics.number_types) or \
                                       (hasattr(eps_infinity,'__len__') and len(eps_infinity)==3)) and \
                                      (isinstance(mu_infinity,numerics.number_types) or \
                                       (hasattr(mu_infinity,'__len__') and len(mu_infinity)==3))),\
                              exception=ValueError)
        
        self.eps_infinity=eps_infinity
        self.mu_infinity=mu_infinity
        
        self.diag_into_matrix=numpy.vectorize(self.diag_into_matrix,otypes=[object])
        
    @staticmethod
    def diag_into_matrix(diag_x,diag_y,diag_z):
            
        
        zeros=diag_x*0
        return numpy.matrix([[diag_x, zeros,      zeros],\
                             [zeros,    diag_y,   zeros],\
                             [zeros,    zeros,      diag_z]]).astype('complex')
            
        
    def epsilon_anisotropic(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        principle_epsilons=self.eps_infinity
        
        return self.diag_into_matrix(*principle_epsilons)
        
    def mu_anisotropic(self,freq,q=None):
        """Anistropic mu tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        principle_mus=self.mu_infinity
        
        return self.diag_into_matrix(*principle_mus)
    
    @staticmethod
    def average_diagonal(matrix):
        
        ##Take average of diagonal entries (is there a more appropriate way?)##
        def extract_average(matrix):
            
            diag=numpy.diag(matrix.astype(complex))
            return numpy.mean(diag)
        
        #We'll only vectorize this if we received an array of anisotropic
        #tensors, we don't want to vectorize over a single tensor, since 
        #that would screw things up.
        if not isinstance(matrix,numpy.matrix):
            extract_average=numpy.vectorize(extract_average,otypes=[complex])
        
        return extract_average(matrix)
    
    def epsilon(self,freq,q=None):
        """Right now this is only implemented to return the "effective"
        epsilon (i.e. ordinary part) inside the material which enters into
        the uniaxial expression for rp."""
        
        #eps_anisotropic=self.epsilon_anisotropic(freq, q)
        #result=self.average_diagonal(eps_anisotropic)
        
        return self.ordinary_epsilon(freq,q)
    
    def mu(self,freq,q=None):
        
        mu_anisotropic=self.mu_anisotropic(freq, q)
        result=self.average_diagonal(mu_anisotropic)
        
        return ensure_complex(result)
    
    @staticmethod
    def extraordinary_component(matrix): #extraordinary means parallel to optical axis
        
        ##Take c-axis entry##
        def get_extraordinary(matrix):
            
            diag=numpy.diag(matrix.astype(complex))
            return diag[2]
        
        #We'll only vectorize this if we received an array of anisotropic
        #tensors, we don't want to vectorize over a single tensor, since 
        #that would screw things up.
        if not isinstance(matrix,numpy.matrix):
            get_extraordinary=numpy.vectorize(get_extraordinary,otypes=[complex])
            
        return get_extraordinary(matrix)
    
    @staticmethod
    def ordinary_component(matrix): #ordinary means perpendicular to optical axis
        
        ##Take average of x&y entries (is there a more appropriate way?)##
        def get_ordinary(matrix):
            
            diag=numpy.diag(matrix.astype(complex))
            return numpy.mean(diag[:2])
        
        #We'll only vectorize this if we received an array of anisotropic
        #tensors, we don't want to vectorize over a single tensor, since 
        #that would improperly use the vectorized operation
        if not isinstance(matrix,numpy.matrix):
            get_ordinary=numpy.vectorize(get_ordinary,otypes=[complex])
            
        return get_ordinary(matrix)
    
    def ordinary_epsilon(self,freq,q=None):
        
        eps_anisotropic=self.epsilon_anisotropic(freq, q)
        result=self.ordinary_component(eps_anisotropic)
        
        return ensure_complex(result)
    
    def extraordinary_epsilon(self,freq,q=None):
        
        eps_anisotropic=self.epsilon_anisotropic(freq, q)
        result=self.extraordinary_component(eps_anisotropic)
        
        return ensure_complex(result)
    
    def ordinary_mu(self,freq,q=None):
        
        mu_anisotropic=self.mu_anisotropic(freq, q)
        result=self.ordinary_component(mu_anisotropic)
        
        return ensure_complex(result)
    
    def extraordinary_mu(self,freq,q=None):
        
        mu_anisotropic=self.mu_anisotropic(freq, q)
        result=self.extraordinary_component(mu_anisotropic)
        
        return ensure_complex(result)
    
    def ordinary_optical_constants(self,freq,q=None):
        
        ordinary_eps=self.ordinary_epsilon(freq,q)
        ordinary_mu=self.ordinary_mu(freq,q)
        
        return safe_sqrt(ordinary_eps*ordinary_mu)
    
    def extraordinary_optical_constants(self,freq,q=None):
        
        extraordinary_eps=self.extraordinary_epsilon(freq,q)
        extraordinary_mu=self.extraordinary_mu(freq,q)
        
        return safe_sqrt(extraordinary_eps*extraordinary_mu)
    
    def get_extraordinary_kz(self,freq,q=0):
        """Right now this is only implemented to return the "effective"
        kz inside the material which enters into the uniaxial expression
        for rp."""
        
        eps2_o=self.ordinary_epsilon(freq,q)
        eps2_e=self.extraordinary_epsilon(freq,q)
        
        omega=2*pi*freq
        
        kz=safe_sqrt(eps2_o*omega**2-eps2_o/eps2_e*q**2)
        
        return ensure_complex(kz)
    
    def get_ordinary_kz(self,freq,q=0): 
        
        eps_ordinary=self.ordinary_epsilon(freq,q)
        
        omega=2*pi*freq
        kz=safe_sqrt(eps_ordinary*omega**2-q**2)
        
        return ensure_complex(kz)
    
    def get_kz(self,freq,q=0):
        """Right now this is only implemented to return the "effective"
        kz inside the material which enters into the uniaxial expression
        for rp."""
        
        return self.get_extraordinary_kz(freq,q)
    
    def reflection_s(self,freq,q=0,angle=None,\
                     entrance=None,entrance_kz=None,\
                     surface=None,**kwargs):
        """Assumes c-axis of uniaxial crystal is perpendicular to reflection plane.
        TODO: generalize."""
        
        if not entrance: entrance=Air
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rs=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        if not entrance_kz: entrance_kz=entrance.get_kz
        ki=entrance_kz(freq,q)
        ko=self.get_ordinary_kz(freq,q)
        
        ##Don't know how to generalize this right now
        #omega=2*pi*freq
        #if surface:
        #    if hasattr(surface,'__len__'):
        #        sigma=numpy.sum([surface.conductivity(freq) for \
        #                         surface in surface], axis=0)
        #    else: sigma=surface.conductivity(freq)
        #    surf=4*pi*kz1*kz2*sigma/(c*omega)
        #else: surf=0
        surf=0
        
        rs+=(ki-ko+surf)/\
            (ki+ko+surf)
               
        return ensure_complex(rs)
    
    def reflection_p(self,freq,q=0,angle=None,\
                     entrance=None,entrance_kz=None,\
                     surface=None,screening=1,**kwargs):
        """Assumes c-axis of uniaxial crystal is perpendicular to reflection plane.
        @TODO: generalize."""
        
        if entrance is None: entrance=Air
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        eps1=entrance.epsilon(freq,q)
        eps2_o=self.ordinary_epsilon(freq,q)*screening
        
        if entrance_kz is None: entrance_kz=entrance.get_kz
        ki=entrance_kz(freq,q)
        ke=self.get_kz(freq,q)
        num=eps2_o*ki-eps1*ke
        den=eps2_o*ki+eps1*ke
        
        omega=2*pi*freq
        
        #Don't know how to generalize this right now
        if surface:
            if hasattr(surface,'__len__'):
                sigma=numpy.sum([surface.conductivity(freq) for \
                                 surface in surface], axis=0)
            else: sigma=surface.conductivity(freq)
            surf=4*pi*ke*ki*sigma/(c*omega)
        else: surf=0
        
        rp+=(num+surf)/\
            (den+surf)
               
        return ensure_complex(rp)
    
class AnisotropicMaterial(BaseAnisotropicMaterial,IsotropicMaterial):
    
    def __init__(self,eps_infinity=1,mu_infinity=1,\
                 drude_params=[],\
                 phonon_params=[],\
                 eps_lps=[],mu_lps=[],\
                 eps_vps=[],mu_vps=[]):
        
        ##Process drude params##
        if len(drude_params):
            if not hasattr(drude_params[0],'__len__'): drude_params=[drude_params]*3
            Logger.raiseException('*drude_params* should be either a Drude parameter set or '+\
                                  'a list of three Drude parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(drude_params)==3), exception=ValueError)
        else: drude_params=[[]]*3
        self.drude_params=drude_params
        
        ##Process phonon params##
        if len(phonon_params):
            if not hasattr(phonon_params[0][0],'__len__'): phonon_params=[phonon_params]*3
            Logger.raiseException('*phonon_params* should be either a phonon parameter set (list of phonon tuples) or '+\
                                  'a list of three phonon parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(phonon_params)==3), exception=ValueError)
        else: phonon_params=[[]]*3
        self.phonon_params=phonon_params
            
        ##Process lorentzian params
        if len(eps_lps):
            if not hasattr(eps_lps[0],'__len__'): eps_lps=[eps_lps]*3
            Logger.raiseException('*eps_lps* should be either a lorentzian parameter set or '+\
                                  'a list of three lorentzian parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(eps_lps)==3), exception=ValueError)
        else: eps_lps=[[]]*3
        if len(mu_lps):
            if not hasattr(mu_lps[0],'__len__'): mu_lps=[mu_lps]*3
            Logger.raiseException('*mu_lps* should be either a lorentzian parameter set or '+\
                                  'a list of three lorentzian parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(mu_lps)==3), exception=ValueError)
        else: mu_lps=[[]]*3
        self.eps_lps=eps_lps
        self.mu_lps=mu_lps
        
        ##Process voigt params##
        if len(eps_vps):
            if not hasattr(eps_vps[0],'__len__'): eps_vps=[eps_vps]*3
            Logger.raiseException('*eps_vps* should be either a Voigt parameter set or '+\
                                  'a list of three Voigt parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(eps_vps)==3), exception=ValueError)
        else: eps_vps=[[]]*3
        if len(mu_vps):
            if not hasattr(mu_vps[0],'__len__'): mu_vps=[mu_vps]*3
            Logger.raiseException('*mu_vps* should be either a Voigt parameter set or '+\
                                  'a list of three Voigt parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(mu_vps)==3), exception=ValueError)
        else: mu_vps=[[]]*3
        self.eps_vps=eps_vps
        self.mu_vps=mu_vps
        
        BaseAnisotropicMaterial.__init__(self,eps_infinity,mu_infinity)
    
    def invoke_property_function(self,freq,q,prop_func,props):
        
        if isinstance(prop_func,str): prop_func=getattr(self,prop_func)
        
        #props should be made into a list of property names
        if isinstance(props,str) or not hasattr(props,'__len__'): props=[props]
        else: props=list(props)
        
        #Request the corresponding attribute for each property in the prop list#
        for i,property in enumerate(props):
            if isinstance(property,str): props[i]=getattr(self,property)
            
        #Logic of the next step is as follows:
        #    Should have *props* as a list of *[3-tuple: parameter 1, 3-tuple: parameter 2, ...]*
        #    We want instead *[(a-axis parameter 1, a-axis parameter 2, ...),
        #                      (b-axis parameter 1, b-axis parameter 2, ...),
        #                      (c-axis parameter 1, c-axis parameter 2, ...)]*
        return [prop_func(freq,q,*axis_props) \
                for axis_props in zip(*props)]
        
    def epsilon_anisotropic(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        eps=copy.copy(self.eps_infinity)
        for prop,prop_func in self.eps_property_functions.items():
             eps_addition=self.invoke_property_function(freq,q,prop,prop_func)
             for i in range(3):
                 eps[i]+=eps_addition[i]
        
        return self.diag_into_matrix(*eps)
        
    def mu_anisotropic(self,freq,q=None):
        """Anistropic mu tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        mu=copy.copy(self.mu_infinity)
        for prop,prop_func in self.mu_property_functions.items():
             mu_addition=self.invoke_property_function(freq,q,prop,prop_func)
             for i in range(3):
                 mu[i]+=mu_addition[i]
        
        return self.diag_into_matrix(*mu)

class TabulatedAnisotropicMaterial(BaseAnisotropicMaterial):
    
    def __init__(self,ordinary_eps_data,extraordinary_eps_data,factor=1):
        
        self._ordinary_eps_data=ordinary_eps_data
        self._extraordinary_eps_data=extraordinary_eps_data
        self.factor=factor
        self.bounds_error=False
        self.extrapolate=True
        
        BaseAnisotropicMaterial.__init__(self)
        
    def ordinary_epsilon(self,freq,q=0,**kwargs):
        
        return self.factor*\
                self._ordinary_eps_data.interpolate_axis(freq,axis=0,
                                                         bounds_error=self.bounds_error,
                                                         extrapolate=self.extrapolate)
        
    def extraordinary_epsilon(self,freq,q=0,**kwargs):
        
        return self.factor*\
                self._extraordinary_eps_data.interpolate_axis(freq,axis=0,
                                                              bounds_error=self.bounds_error,
                                                              extrapolate=self.extrapolate)

class TabulatedAnisotropicMaterialFromFile(TabulatedAnisotropicMaterial):
    
    data_dir=os.path.join(os.path.dirname(__file__),\
                          'Tabulated')
    
    def __init__(self,epsfile,factor=1):
        
        if not os.path.exists(epsfile):
            epsfile=os.path.join(self.data_dir,epsfile)
        Logger.write('Loading tabulated material data from file "%s"...'%epsfile)
        file=open(epsfile)
        
        if epsfile.lower().endswith('.pickle'):
            file=open(epsfile,'rb')
            eps_data=pickle.load(file,encoding='bytes')
            ordinary_eps_data=eps_data['ordinary']
            extraordinary_eps_data=eps_data['extraordinary']
        elif epsfile.lower().endswith('.csv'):
            freq,ordinary_eps1,ordinary_eps2,\
                extraordinary_eps1,extraordinary_eps2=misc.extract_array(file, dtype=numpy.float).T
            ordinary_eps_data=AWA(ordinary_eps1+1j*ordinary_eps2,\
                                  axes=[freq],axis_names=['Frequency [cm^-1]'])
            extraordinary_eps_data=AWA(extraordinary_eps1+1j*extraordinary_eps2,\
                                       axes=[freq],axis_names=['Frequency [cm^-1]'])
        else: Logger.error('File type not understood for file "%s".'%epsfile)
        
        file.close()
        
        TabulatedAnisotropicMaterial.__init__(self,ordinary_eps_data,extraordinary_eps_data,factor=factor)

class DispersiveAnisotropicMaterial(AnisotropicMaterial):
    """This was originally an implementation meant for numerical
    experiments to study the effect of phonon dispersion in the
    Brillouin zone on near-field spectra.
    
    The conclusion:
    Near-field momenta are not sufficiently
    high to probe this regime, there is no clear
    effect on near-field spectra.
    """
    
    def __init__(self,eps_infinity=1,mu_infinity=1,\
                 drude_params=[],\
                 eps_lps=[],mu_lps=[],\
                 eps_vps=[],mu_vps=[],\
                 disperse_by=-30,\
                 lattice_constant=.3e-7):
        
        self._disperse_by=disperse_by
        self._lattice_constant=lattice_constant
        AnisotropicMaterial.__init__(self,eps_infinity=eps_infinity,mu_infinity=mu_infinity,\
                                     drude_params=drude_params,\
                                     eps_lps=eps_lps,mu_lps=mu_lps,\
                                     eps_vps=eps_vps,mu_vps=mu_vps)
        
    def get_lorentzians(self,freq,q,lps):
        
        if len(lps) and not hasattr(lps[0],'__len__'): lps=[lps]
        
        #Get characteristic dispersion values#
        lattice_constant=self._lattice_constant
        qmax=2*pi/(2*lattice_constant)
        qcenter=qmax/2.
        dq=qmax/4.
        
        result=0
        for L in lps:
            amp,resonance,damping=L
            
            #Disperse the resonant frequency by an envelope in q#
            disperse_by=self._disperse_by
            fraction_to_disperse=disperse_by/float(resonance)
            resonance=resonance*(1+fraction_to_disperse/2.*\
                                   (numpy.tanh((q-qcenter)/dq)+1))
            
            result+=amp*resonance*damping/(resonance**2-freq**2-1j*damping*freq)
        
        return result
    
def compute_Xfer_matrix(freq,q,media=[],thicknesses=[]):
    """Thicknesses in cm."""
    
    if not hasattr(media,'__len__'): media=[media]
    if not hasattr(media,'__len__'): thicknesses=[thicknesses]
    Logger.raiseException('*media* and *thicknesses* must be of the same length.',\
                          unless=(len(media)==len(thicknesses)),\
                          exception=IndexError)
    
    ##Define a function that will return the transfer matrix for one of the media
    #at a given freq/q##
    def get_Xfer_matrix(freq,q,medium,thickness):
        
        kz=2*pi*medium.get_kz(freq,q)
        return numpy.matrix([[numpy.cos(kz*thickness),      1/kz*numpy.sin(kz*thickness)],\
                             [-kz*numpy.sin(kz*thickness),  numpy.cos(kz*thickness)]])
        
    get_Xfer_matrix=numpy.vectorize(get_Xfer_matrix,\
                                    otypes=[object])
    
    ##Compute Xfer matrices (or rather, arrays of Xfer matrices - one for each freq/q)##
    Xfer_matrices=[get_Xfer_matrix(freq,q,medium,thickness) for \
                   medium,thickness in zip(media,thicknesses)]
    
    #Reverse the order, as is applicable for layers N...1 before multiplication.
    Xfer_matrices.reverse()
    
    ##Define a function that will multiply transfer matrices
    #at a given freq/q##
    #We use *matrices since we may want to vectorize over array arguments,
    #and we wouldn't want to confuse this with vectorizing over what would
    #otherwise be a list of matrices (or arrays of matrices).
    def multiply_Xfer_matrices(*matrices):
        
        product=numpy.matrix(numpy.eye(2))
        for matrix in matrices:
            product=product*matrix
            
        return product
    
    multiply_Xfer_matrices=numpy.vectorize(multiply_Xfer_matrices,\
                                           otypes=[object])
    
    ##Compute the total Xfer matrix##
    return multiply_Xfer_matrices(*Xfer_matrices)
    
def MultilayerCoefficients(freq,q,media=[],thicknesses=[],entrance=Air,exit=Air):
    """Thicknesses in cm."""
    
    L=numpy.sum(thicknesses)
    M=compute_Xfer_matrix(freq,q,media,thicknesses)
    
    #Define a function to extract an element of a Xfer matrix
    #for a given freq/q
    def extract_element(i,j,matrix): return matrix[i,j]
    
    #We'll only vectorize this if we received an array of matrices,
    #we don't want to vectorize over a single matrix, since 
    #that would screw things up.
    if not isinstance(M,numpy.matrix):
        extract_element=numpy.vectorize(extract_element,otypes=[complex])
        
    #Turn to complex type,
    M11=extract_element(0,0,M)
    M12=extract_element(0,1,M)
    M21=extract_element(1,0,M)
    M22=extract_element(1,1,M)
    
    kzL=2*pi*entrance.get_kz(freq,q)
    kzR=2*pi*exit.get_kz(freq,q)
    
    #Must make all arrays complex data type, "object" won't properly multiply float128
    if isinstance(kzL,numpy.ndarray):
        kzL=kzL.astype(complex)
        kzR=kzR.astype(complex)
    
    r=((M21+kzL*kzR*M12)+1j*(kzL*M22-kzR*M11))/\
      ((-M21+kzL*kzR*M12)+1j*(kzL*M22+kzR*M11))
      
    t=2j*kzL*numpy.exp(-1j*kzR*L)*(M11*M22-M12*M21)/\
                                  (-M21+kzL*kzR*M12+1j*(kzR*M11+kzL*M22))
    
    return r,t

class BaseUniaxialMaterial(object):
    
    def __init__(self,eps_infinity=1):
        
        if isinstance(eps_infinity,numerics.number_types): eps_infinity=[eps_infinity]*2
        Logger.raiseException('*eps_infinity* should be either a number, or an '+\
                              'iterable of length two (one constant each for the '+\
                              'ordinary and extraordinary optical axes, respectively).',\
                              unless=(isinstance(eps_infinity,numerics.number_types) or \
                                       (hasattr(eps_infinity,'__len__') and len(eps_infinity)==2)),\
                              exception=ValueError)
        
        self.eps_infinity=eps_infinity
        
        self.set_rotation(0)
        
    def set_rotation(self,angle,about_axis='z'):
        """Reset extraordinary axis to z-axis, then rotate the extraordinary
        axis by some `angle` (in degrees) about
        a 3-tuple axis `about_axis` (can be one of 'x','y', or 'z')"""
        
        #Reset the rotation matrix and extraordinary axis
        self._R=numpy.matrix(numpy.eye(3))
        self._a=numpy.matrix([0,0,1]).T
        
        self.rotate(angle,about_axis=about_axis)
        
    def rotate(self,angle,about_axis='z'):
        """Rotate the extraordinary axis by some `angle` (in degrees) about
        a 3-tuple axis `about_axis` (can be one of 'x','y', or 'z')"""
        
        if about_axis=='x': about_axis=[1,0,0]
        elif about_axis=='y': about_axis=[0,1,0]
        elif about_axis=='z': about_axis=[0,0,1]
        
        r=numerics.rotation_matrix(angle,about_axis)
        self._R=r*self._R
        self._a=r*self._a
        
    def get_rotation(self): return copy.copy(self._R)
    
    def get_aVec(self): return copy.copy(self._a)
            
    def epsilon_principle(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        eps_axes=self.eps_infinity
        
        return anisotropic.elements_into_diag_3x3matrices(*[eps_axes[0],eps_axes[0],eps_axes[1]])
    
    def epsilon_anisotropic(self,freq,q=None):
        
        return anisotropic.vectorize_matrix_func(lambda eps,R: R*eps*R.I)\
                                                (self.epsilon_principle(freq,q),\
                                                 self.get_rotation())
    
    def epsilon_uniaxial_components(self,freq,q=None):
        """Returns ordinary and "primed" components of the
        uniaxial dielectric tensor."""
        
        eps=self.epsilon_principle(freq,q=q)
        
        #Get the ordinary component
        eps_o=anisotropic.elements_from_3x3matrices(eps,'xx')
        eps_e=anisotropic.elements_from_3x3matrices(eps,'zz')
        eps_p=eps_e-eps_o
        
        return eps_o,eps_p
    
    def epsilon(self,freq,q=None,element=None):
        
        eps=self.epsilon_anisotropic(freq,q)
        if element:
            eps=anisotropic.elements_from_3x3matrices(eps,element)
        
        return eps
        
    def get_kz_transverse(self,freq,q,QS=False,eps_o=None):
        """For the transverse ray traveling through the crystal, possessing
        a purely transverse component to the polarization."""
        
        if QS: return 1j*2*numpy.pi*freq
        
        if eps_o is None or (not hasattr(freq,'__len__') or \
                             len(eps_o)==len(freq)):
            eps_o,eps_p=self.epsilon_uniaxial_components(freq, q)
            
        kz_o=safe_sqrt(2*numpy.pi*freq*eps_o-q**2)
        
        return ensure_complex(kz_o)
    
    def get_kz_mixed(self,freq,q,QS=False,eps_o=None,eps_p=None):
        """For the mixed ray traveling through the crystal, possessing
        both a transverse and longitudinal component to the polarization.
        
        Eq. (29) of Ignatovitch & Ignatovitch 2011"""
        
        if QS: return 1j*2*numpy.pi*freq
        
        if eps_o is None or eps_p is None or\
            (not hasattr(freq,'__len__') or \
             (len(eps_o)==len(freq) and len(eps_p)==len(freq))):
            eps_o,eps_p=self.epsilon_uniaxial_components(freq, q)
            
        aVec=self.get_aVec()
        nDotA=anisotropic.element_from_3vectors(aVec,'z')
        lDotA=anisotropic.element_from_3vectors(aVec,'x')
        
        k0=2*numpy.pi*freq
        eta=eps_p/eps_o
        
        num=-eta*q*nDotA*lDotA+safe_sqrt(eps_o*k0*(1+eta)*(1+eta*nDotA**2)-q**2*(1+eta*lDotA**2+eta*nDotA**2))
        den=1+eta*nDotA**2
        
        return ensure_complex(num/den)
    
    get_kz=get_kz_transverse
    
    #Magnetic materials not implemented within scope of this class
    def mu(self,freq,q=None): return 1
    
    def get_E_transverse_polarization(self,kVec):
        
        from numpy.linalg import norm
        def mag(vec): return safe_sqrt(numpy.sum(numpy.array(vec).squeeze()**2,axis=-1))
        
        aVec=self.get_aVec()
        aCross=numerics.cross_product_matrix(aVec)
        
        #Make sure cross product first brings kvec to unit vector
        Efunc=lambda kVec: aCross*(kVec/mag(kVec))
        
        EVec=anisotropic.vectorize_matrix_func(Efunc)(kVec)
        
        return anisotropic.vectorize_matrix_func(lambda vec: vec/norm(vec))(EVec)
    
    def get_E_mixed_polarization(self,kVec,eps_o,eps_p):
        """`kvec` is expected to be a 3-vector (matrix) or an array of 
        such 3-vectors, with `eps_o` and `eps_o` complex numbers, or
        arrays of the same length."""
        
        from numpy.linalg import norm
        def mag(vec): return safe_sqrt(numpy.sum(numpy.array(vec).squeeze()**2,axis=-1))
        
        aVec=self.get_aVec()
        eta=eps_p/eps_o
        
        normVec=lambda vec: vec/norm(vec)
        kappaVec=anisotropic.vectorize_matrix_func(normVec)(kVec)
        
        dotA=lambda kappaVec: aVec.T*kappaVec
        Efunc=lambda kappaVec,eta: aVec-kappaVec*(dotA(kappaVec)*(1+eta))/\
                                                 (1+eta*dotA(kappaVec)**2)
        
        EVec=anisotropic.vectorize_matrix_func(Efunc)(kappaVec,eta)
        
        return anisotropic.vectorize_matrix_func(lambda vec: vec/norm(vec))(EVec)
    
    def get_H_polarization(self,freq,kVec,EVec):
        
        Hfunc=lambda freq,kVec,EVec: numerics.cross_product_matrix(kVec)*EVec/(2*numpy.pi*freq)
        
        HVec=anisotropic.vectorize_matrix_func(Hfunc)(freq,kVec,EVec)
        
        #Do not normalize- H must be scaled properly with respect to the associated E-field
        return HVec#anisotropic.vectorize_matrix_func(lambda vec: vec/norm(vec))(HVec)
    
    def reflection_p(self,freq,q=0,angle=None,entrance=Air,surface=None,QS=False,**kwargs):
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        #In our coordinates, unit vectors from Ignatovitch & Ignatovitch 2011 are as follows:
        l='x'; t='y'; n='z'
        eps_o,eps_p=self.epsilon_uniaxial_components(freq,q)
        
        #It is unclear whether these are the only parameters needed from entrance material
        #(where is need for epsilon in incident material?)
        k0_inc=2*numpy.pi*freq*entrance.epsilon(freq,q)
        kz_inc=entrance.get_kz(freq,q)
        kappaIncPerp=kz_inc/k0_inc
        
        #Get propogation vectors and fields within the medium
        kz_trans=self.get_kz_transverse(freq, q=q, QS=QS, eps_o=eps_o)
        kVecTrans=anisotropic.elements_into_3vectors(q,0,kz_trans)
        self.kVecTrans=kVecTrans
        EVecTrans=self.get_E_transverse_polarization(kVecTrans)
        HVecTrans=self.get_H_polarization(freq, kVecTrans, EVecTrans)
        
        kz_mixed=self.get_kz_mixed(freq, q=q, QS=QS, eps_o=eps_o, eps_p=eps_p)
        kVecMixed=anisotropic.elements_into_3vectors(q,0,kz_mixed)
        self.kVecMixed=kVecMixed
        EVecMixed=self.get_E_mixed_polarization(kVecMixed, eps_o, eps_p)
        HVecMixed=self.get_H_polarization(freq, kVecMixed, EVecMixed)
        
        #Get desired components of fields
        tE1=anisotropic.elements_from_3vectors(EVecTrans,t)
        lE1=anisotropic.elements_from_3vectors(EVecTrans,l)
        tH1=anisotropic.elements_from_3vectors(HVecTrans,t)
        lH1=anisotropic.elements_from_3vectors(HVecTrans,l)
        
        tE2=anisotropic.elements_from_3vectors(EVecMixed,t)
        lE2=anisotropic.elements_from_3vectors(EVecMixed,l)
        tH2=anisotropic.elements_from_3vectors(HVecMixed,t)
        lH2=anisotropic.elements_from_3vectors(HVecMixed,l)
        
        #Proceed to get transmission coefficients
        inhomVec=anisotropic.elements_into_2vectors(2*kappaIncPerp,0)
        MultMatrix=anisotropic.elements_into_2x2matrices(kappaIncPerp*tH1+lE1,kappaIncPerp*tH2+lE2,\
                                                         kappaIncPerp*tE1-lH1,kappaIncPerp*tE2-lH2)
        tauVec=anisotropic.vectorize_matrix_func(lambda MultMatrix,inhomVec: MultMatrix.I*inhomVec)\
                                                    (MultMatrix,inhomVec)
        self.tau1=anisotropic.elements_from_2vectors(tauVec,'x')
        self.tau2=anisotropic.elements_from_2vectors(tauVec,'y')
                                                
        #Proceed to get reflection coefficients
        inhomVec=numpy.matrix([[0],[1]])
        MultMatrix=anisotropic.elements_into_2x2matrices(tE1,tE2,\
                                                        tH1,tH2)
        rhoVec=anisotropic.vectorize_matrix_func(lambda MultMatrix,tauVec: MultMatrix*tauVec-inhomVec)\
                                                    (MultMatrix,tauVec)
        
        #First component is TM-->TE component, second is TM-->TM
        rp+=anisotropic.elements_from_2vectors(rhoVec,'y')
        return rp

class UniaxialMaterial(BaseUniaxialMaterial,IsotropicMaterial):
    
    def __init__(self,eps_infinity=1,mu_infinity=1,\
                 drude_params=[],\
                 phonon_params=[],\
                 eps_lps=[],mu_lps=[],\
                 eps_vps=[],mu_vps=[]):
        
        ##Process drude params##
        if len(drude_params):
            if not hasattr(drude_params[0],'__len__'): drude_params=[drude_params]*2
            Logger.raiseException('*drude_params* should be either a Drude parameter set or '+\
                                  'a list of two Drude parameter sets (for the ordinary and '+\
                                  'extraordinary optical components, respectively).',\
                                  unless=(len(drude_params)==2), exception=ValueError)
        else: drude_params=[[]]*2
        self.drude_params=drude_params
        
        ##Process phonon params##
        if len(phonon_params):
            if not hasattr(phonon_params[0][0],'__len__'): phonon_params=[phonon_params]*3
            Logger.raiseException('*phonon_params* should be either a phonon parameter set (list of phonon tuples) or '+\
                                  'a list of two phonon parameter sets (for the ordinary and '+\
                                  'extraordinary optical components, respectively).',\
                                  unless=(len(phonon_params)==2), exception=ValueError)
        else: phonon_params=[[]]*2
        self.phonon_params=phonon_params
            
        ##Process lorentzian params
        if len(eps_lps):
            if not hasattr(eps_lps[0],'__len__'): eps_lps=[eps_lps]*2
            Logger.raiseException('*eps_lps* should be either a lorentzian parameter set or '+\
                                  'a list of two lorentzian parameter sets (for the ordinary and '+\
                                  'extraordinary optical components, respectively).',\
                                  unless=(len(eps_lps)==2), exception=ValueError)
        else: eps_lps=[[]]*2
        self.eps_lps=eps_lps
        
        ##Process voigt params##
        if len(eps_vps):
            if not hasattr(eps_vps[0],'__len__'): eps_vps=[eps_vps]*2
            Logger.raiseException('*eps_vps* should be either a Voigt parameter set or '+\
                                  'a list of two Voigt parameter sets (for the ordinary and '+\
                                  'extraordinary optical components, respectively).',\
                                  unless=(len(eps_vps)==2), exception=ValueError)
        else: eps_vps=[[]]*2
        self.eps_vps=eps_vps
        
        BaseUniaxialMaterial.__init__(self,eps_infinity)
    
    def invoke_property_function(self,freq,q,prop_func,props):
        
        if isinstance(prop_func,str): prop_func=getattr(self,prop_func)
        
        #props should be made into a list of property names
        if isinstance(props,str) or not hasattr(props,'__len__'): props=[props]
        else: props=list(props)
        
        #Request the corresponding attribute for each property in the prop list#
        for i,property in enumerate(props):
            if isinstance(property,str): props[i]=getattr(self,property)
            
        #Logic of the next step is as follows:
        #    Should have *props* as a list of *[3-tuple: parameter 1, 3-tuple: parameter 2, ...]*
        #    We want instead *[(a-axis parameter 1, a-axis parameter 2, ...),
        #                      (b-axis parameter 1, b-axis parameter 2, ...),
        #                      (c-axis parameter 1, c-axis parameter 2, ...)]*
        return [prop_func(freq,q,*axis_props) \
                for axis_props in zip(*props)]
        
    def epsilon_principle(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the extraordinary axis of the crystal."""
        
        eps=copy.copy(self.eps_infinity)
        for prop,prop_func in self.eps_property_functions.items():
             eps_addition=self.invoke_property_function(freq,q,prop,prop_func)
             for i in range(2):
                 eps[i]+=eps_addition[i]
        
        return anisotropic.elements_into_diag_3x3matrices(*[eps[0],eps[0],eps[1]])


class BaseAnisotropicMaterial2(object):
    
    def __init__(self,eps_infinity=1,mu_infinity=1):
        
        if isinstance(eps_infinity,numerics.number_types): eps_infinity=[eps_infinity]*3
        if isinstance(mu_infinity,numerics.number_types): mu_infinity=[mu_infinity]*3
        Logger.raiseException('*eps_infinity* and *mu_infinity* should be either numbers, '+\
                              'or iterables of length three (one constant for each of the '+\
                              'principle axes, respectively).',\
                              unless=((isinstance(eps_infinity,numerics.number_types) or \
                                       (hasattr(eps_infinity,'__len__') and len(eps_infinity)==3)) and \
                                      (isinstance(mu_infinity,numerics.number_types) or \
                                       (hasattr(mu_infinity,'__len__') and len(mu_infinity)==3))),\
                              exception=ValueError)
        
        self.eps_infinity=eps_infinity
        self.mu_infinity=mu_infinity
        
        self.set_rotation(0,0,0)
        
    def set_rotation(self,anglex=0,angley=0,anglez=0,reset=True):
        """Rotate around each axis in sequence."""
        
        if reset:
            axes_col_vecs=numpy.matrix(numpy.eye(3))
            R=numpy.matrix(numpy.eye(3))
        else:
            axes_col_vecs=numpy.matrix(numpy.array(self.get_axes()).T)
            R=self._R
        
        fixed_col_vecs=numpy.matrix(numpy.eye(3))
        for i,angle in enumerate((anglex,angley,anglez)):
            if angle==0: continue
            about_axis=numpy.squeeze(numpy.array(fixed_col_vecs[:,i]))
            Raxis=numerics.rotation_matrix(angle,about_axis)
            axes_col_vecs*=Raxis
            R*=Raxis
            
        self._R=R
        axes_col_vecs=numpy.array(axes_col_vecs)
        self._axes=list(axes_col_vecs.T)
        
    def get_rotation(self): return self._R.copy()
    
    def get_axes(self): return [axis.copy() for axis in self._axes]
        
    @classmethod
    def optimize_kz(cls,freq,q,eps,mu):
        
        omega=2*pi*freq
        kz0=safe_sqrt(eps[2,2]*freq**2-q**2)
        kz0=[kz0.real,kz0.imag]
        
        #Powell's method works nicely, a little faster than fmin
        #result=scipy.optimize.fmin_powell(anisotropic.detO,\
        #                                  x0=kz0,args=(q,omega,eps,mu),\
        #                                  disp=False,\
        #                                  ftol=1e-10,\
        #                                  xtol=1e-10)
        
        #The best, and fastest - not sure why, but factor=.1 is crucial
        #Also, requires that `anisotropic.detO` output a tuple (e.g. `real,imag`)
        result=scipy.optimize.fsolve(anisotropic.detO,\
                                          x0=kz0,args=(q,omega,eps,mu),\
                                          factor=.1,xtol=1e-12)
        
        #Fmin works nicely, but its behavior is not well understood
        #http://docs.scipy.org/doc/scipy-0.10.0/reference/generated/scipy.optimize.fmin.html#scipy.optimize.fmin
        #result=scipy.optimize.fmin(anisotropic.detO,\
        #                           x0=kz0,args=(q,omega,eps,mu),\
        #                           disp=False)
        
        #Newton-Conjugate-Gradient method and related require gradient estimate,
        #they get stuck at kz0
        #http://docs.scipy.org/doc/scipy-0.10.0/reference/generated/scipy.optimize.fmin_ncg.html#scipy.optimize.fmin_ncg
        #result=scipy.optimize.fmin_ncg(anisotropic.detO,\
        #                                fprime=anisotropic.detO_gradient,\
        #                                x0=kz0,args=(q,omega,eps,mu),\
        #                                disp=False)
        
        kz=result[0]+\
           result[1]*1j
       
        return kz
        
    def epsilon_principle(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        eps_axes=self.eps_infinity
        
        return anisotropic.elements_into_diag_3x3matrices(*eps_axes)
        
    def mu_principle(self,freq,q=None):
        """Anistropic mu tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        mu_axes=self.mu_infinity
        
        return anisotropic.elements_into_diag_3x3matrices(*mu_axes)
    
    def epsilon_anisotropic(self,freq,q=None):
        
        return anisotropic.vectorize_matrix_func(lambda eps,R: R*eps*R.I)\
                                                (self.epsilon_principle(freq,q),\
                                                 self.get_rotation())
    
    def mu_anisotropic(self,freq,q=None): 
        
        return anisotropic.vectorize_matrix_func(lambda eps,R: R*eps*R.I)\
                                                (self.mu_principle(freq,q),\
                                                 self.get_rotation())
    
    def epsilon(self,freq,q=None,element=None):
        
        eps=self.epsilon_anisotropic(freq,q)
        if element:
            eps=anisotropic.element_from_3x3matrices(eps,element)
        
        return eps
    
    def mu(self,freq,q=None,element=None):
        
        mu=self.mu_anisotropic(freq,q)
        if element:
            mu=anisotropic.element_from_3x3matrices(mu,element)
        
        return mu
        
    def get_kz(self,freq,q=0,eps=None,mu=None,QS=False):
        
        if QS: return 1j*q
        
        eps=self.epsilon(freq,q)
        mu=self.mu(freq,q)
        
        result=anisotropic.vectorize_matrix_func(self.optimize_kz)\
                                                (freq,q,eps,mu)
        
        return result
    
    #---Definitions for matrices anisotropic reflection coefficients
    @staticmethod
    def gamma_matrix(freq,q,epsinv,sigma):
        
        if not sigma: return 0
        
        sigma_matrix=anisotropic.elements_into_3x3matrices(0,0,0,\
                                                           0,0,0,\
                                                           1,0,0)
        
        omega=2*pi*freq
        gamma=4*pi*q*sigma/(omega*c)*anistropic.vectorize_matrix_func(lambda x,y: x*y)\
                                                                    (epsinv,sigma_matrix)
        
        return gamma
        
    @staticmethod
    def delta_matrix(mu,sigma):
        
        if not sigma: return 0
        
        sigma_matrix=anisotropic.elements_into_3x3matrices(0,1,0,\
                                                          -1,0,0,\
                                                           0,0,0)
        
        delta=4*pi*sigma/c*anisotropic.vectorize_matrix_func(lambda x,y: x*y)\
                                                            (mu,sigma_matrix)
        
        return delta
    
    @staticmethod
    def alpha_matrix(eps1,eps2):
        
        n1=anisotropic.vectorize_matrix_func(lambda eps1,z: eps1*z)\
                                            (eps1,numpy.matrix([0,0,1.]).T)
        norm1=anisotropic.vectorize_matrix_func(lambda n1: linalg.norm(n1))\
                                               (n1)
        n2=anisotropic.vectorize_matrix_func(lambda eps2,z: eps2*z)\
                                            (eps2,numpy.matrix([0,0,1.]).T)
        
        I=numpy.matrix(numpy.eye(3))
        
        alpha=anisotropic.vectorize_matrix_func(lambda I,n1,n2,norm1: I+n1*(n2-n1).T/norm1**2)\
                                               (I,n1,n2,norm1)
        
        return alpha
    
    @staticmethod
    def beta_matrix(mu1,mu2_inv):
        
        m1=numpy.matrix(numpy.diag([1,1,0]))
        m2=numpy.matrix(numpy.diag([0,0,1]))
        
        alpha=anisotropic.vectorize_matrix_func(lambda a,b,c,d: a*b*c+d)\
                                               (mu1,m1,mu2_inv,m2)
        
        return alpha
        
    def K_matrix(self,freq,q=0,medium=None,direction=+1,\
                 eps=None,mu=None,kz=None):
        
        if medium is None: medium=self
        if eps is None: eps=medium.epsilon(freq,q)
        if mu is None: mu=medium.mu(freq,q)
        
        if kz is None:
            if isinstance(medium,BaseAnisotropicMaterial2):
                kz=medium.get_kz(freq,q,eps=eps,mu=mu)
            else: kz=medium.get_kz(freq,q)
            
        #kz is assumed in "positive" direction
        omega=2*pi*freq
        return anisotropic.elements_into_3x3matrices(0,             -direction*kz/omega,   0,\
                                                     direction*kz/omega,  0,             -q/omega,\
                                                     0,              q/omega,             0)
        
    def E_matrix(self,freq,q=0,medium=None,direction=+1,\
                 eps=None,mu=None,eps_mu_inv=None,kz=None):
        
        if medium is None: medium=self
        if eps is None: eps=medium.epsilon(freq,q)
        if mu is None: mu=medium.mu(freq,q)
        if eps_mu_inv is None:
            #switch order because of inversion
            mu_eps=anisotropic.vectorize_matrix_func(lambda eps,mu: mu*eps)\
                                                    (eps,mu)
            eps_mu_inv=anisotropic.inv_3x3(mu_eps)
        
        K=self.K_matrix(freq,q,medium=medium,direction=direction,eps=eps,mu=mu,kz=kz)
        
        return -anisotropic.vectorize_matrix_func(lambda eps_mu_inv,K: eps_mu_inv*K)\
                                                 (eps_mu_inv,K)
    
    def B_matrix(self,freq,q=0,medium=None,direction=+1,\
                 eps=None,mu=None,kz=None):
        
        return self.K_matrix(freq,q,medium=medium,direction=direction,eps=eps,mu=mu,kz=kz)
    
    def get_R_matrix(self,freq,q=0,\
                     entrance=Air,surface=None,\
                     eps1=None,mu1=None,kz1=None,\
                     eps2=None,mu2=None,kz2=None):
        
        ##Get optical constants and conductivities##
        if eps1 is None: eps1=entrance.epsilon(freq,q)
        if mu1 is None: mu1=entrance.mu(freq,q)
        if kz1 is None: kz1=entrance.get_kz(freq,q,eps=eps1,mu=mu1)
        
        if eps2 is None: eps2=self.epsilon(freq,q)
        if mu2 is None: mu2=self.mu(freq,q)
        if kz2 is None: kz2=self.get_kz(freq,q,eps=eps2,mu=mu2)
        
        if surface:
            if hasattr(surface,'__len__'):
                sigma=numpy.sum([surface.conductivity(freq) for \
                                 surface in surface], axis=0)
            else: sigma=surface.conductivity(freq)
        else: sigma=0
        
        ##Define matrices I.##
        #Block OK!
        gamma1=0#self.gamma_matrix(freq,q,eps1_inv,sigma)
        alpha1=self.alpha_matrix(eps1,eps2)
        
        #Block OK!
        delta1=0#self.delta_matrix(freq,mu1,sigma)
        mu2_inv=anisotropic.inv_3x3(mu2)
        beta1=self.beta_matrix(mu1,mu2_inv); del mu2_inv
        
        B2down=self.B_matrix(freq,q,medium=self,direction=-1,\
                             eps=eps2, mu=mu2, kz=kz2)
        
        #switch order because of inversion
        mu1_eps1=anisotropic.vectorize_matrix_func(lambda eps,mu: mu*eps)\
                                                (eps1,mu1)
        eps1_mu1_inv=anisotropic.inv_3x3(mu1_eps1)
        E1up=self.E_matrix(freq, q, medium=entrance, direction=+1,\
                           eps=eps1, mu=mu1, eps_mu_inv=eps1_mu1_inv, kz=kz1); del eps1_mu1_inv
                           
        M=anisotropic.vectorize_matrix_func(lambda alpha1,gamma1,E1up,beta1,B2down,delta1: \
                                            alpha1+gamma1-E1up*(beta1*B2down+delta1))\
                                           (alpha1,gamma1,E1up,beta1,B2down,delta1)
        M_inv=anisotropic.inv_3x3(M); del delta1,B2down,M
        
        
        ##Define matrices II.##
        B1down=self.B_matrix(freq,q,medium=entrance,direction=-1,\
                             eps=eps1, mu=mu1, kz=kz1)
        N=anisotropic.vectorize_matrix_func(lambda I,E1up,B1down: \
                                            I-E1up*B1down)\
                                            (numpy.matrix(numpy.eye(3)),E1up,B1down)
        
        ##Get reflection matrix##
        R=anisotropic.vectorize_matrix_func(lambda gamma1,alpha1,M_inv,N,I: \
                                            (gamma1+alpha1)*M_inv*N-I)\
                                           (gamma1,alpha1,M_inv,N,\
                                            numpy.matrix(numpy.eye(3)))
        
        return R
        
    def get_p_Ez(self,freq,q,Ex=1,eps=None,mu=None,kz=None):
        
        if eps is None: eps=self.epsilon(freq,q)
        if kz is None: kz=self.get_kz(freq,q,eps=eps,mu=mu)
        
        eps_xz=anisotropic.element_from_3x3matrices(eps,element='xz')
        eps_zz=anisotropic.element_from_3x3matrices(eps,element='zz')
        eps_xx=anisotropic.element_from_3x3matrices(eps,element='xx')
        eps_zx=anisotropic.element_from_3x3matrices(eps,element='zx')
        
        return -(q*eps_xx+kz*eps_zx)/(q*eps_xz+kz*eps_zz)*Ex
        
    def reflection_p(self,freq,q=0,angle=None,entrance=Air,surface=None,QS=False,**kwargs):
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rp=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        eps1=entrance.epsilon(freq,q)
        mu1=entrance.mu(freq,q)
        kz1=entrance.get_kz(freq,q,eps=eps1,mu=mu1,QS=QS)
        
        eps2=self.epsilon(freq,q)
        mu2=self.mu(freq,q)
        kz2=self.get_kz(freq,q,eps=eps2,mu=mu2,QS=QS)
        
        R_matrix=self.get_R_matrix(freq,q,\
                                     entrance=entrance,surface=surface,\
                                     eps1=eps1,mu1=mu1,kz1=kz1,\
                                     eps2=eps2,mu2=mu2,kz2=kz2)
        
        Ex1=1
        Ez1=entrance.get_p_Ez(freq,q,Ex=Ex1,eps=eps1,mu=mu1,kz=kz1)
        Einc=anisotropic.elements_into_3vectors(Ex1,0,Ez1)
        Eref=anisotropic.vectorize_matrix_func(lambda R,Einc: R*Einc)\
                                              (R_matrix,Einc)
        
        Exref=anisotropic.element_from_3vectors(Eref,'x')
        rp+=Exref/Ex1 #negative since change of E-field convention for inc/ref for conventional rp
        
        return rp
    
    def rotational_reflection_p(self,freq,q=0,angle=None,entrance=Air,surface=None,QS=False,**kwargs):
        
        rp_A=self.reflection_p(freq,q=q,angle=angle,entrance=entrance,surface=surface,QS=QS,**kwargs)
        
        self.set_rotation(anglez=90,reset=False)
        rp_B=self.reflection_p(freq,q=q,angle=angle,entrance=entrance,surface=surface,QS=QS,**kwargs)
        self.set_rotation(anglez=-90,reset=False)
        
        return (rp_A+rp_B)/2.
    
    def reflection_s(self,freq,q=0,angle=None,entrance=Air,surface=None,QS=False,**kwargs):
        
        ##Get holder for data, and expanded freq & q##
        freq,q,rs=_prepare_freq_and_q_holder_(freq,q,\
                                              angle=angle,\
                                              entrance=entrance)
        
        eps1=entrance.epsilon(freq,q)
        mu1=entrance.mu(freq,q)
        kz1=entrance.get_kz(freq,q,eps=eps1,mu=mu1,QS=QS)
        
        eps2=self.epsilon(freq,q)
        mu2=self.mu(freq,q)
        kz2=self.get_kz(freq,q,eps=eps2,mu=mu2,QS=QS)
        
        R_matrix=self.get_R_matrix(freq,q,\
                                     entrance=entrance,surface=surface,\
                                     eps1=eps1,mu1=mu1,kz1=kz1,\
                                     eps2=eps2,mu2=mu2,kz2=kz2)
        
        Ey1=1
        Einc=anisotropic.elements_into_3vectors(0,Ey1,0)
        Eref=anisotropic.vectorize_matrix_func(lambda R,Einc: R*Einc)\
                                              (R_matrix,Einc)
        
        Eyref=anisotropic.element_from_3vectors(Eref,'y')
        rs+=Eyref/Ey1
        
        return rs
        
class AnisotropicMaterial2(BaseAnisotropicMaterial2,IsotropicMaterial):
    
    def __init__(self,eps_infinity=1,mu_infinity=1,\
                 drude_params=[],\
                 phonon_params=[],\
                 eps_lps=[],mu_lps=[],\
                 eps_vps=[],mu_vps=[]):
        
        ##Process drude params##
        if len(drude_params):
            if not hasattr(drude_params[0],'__len__'): drude_params=[drude_params]*3
            Logger.raiseException('*drude_params* should be either a Drude parameter set or '+\
                                  'a list of three Drude parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(drude_params)==3), exception=ValueError)
        else: drude_params=[[]]*3
        self.drude_params=drude_params
        
        ##Process phonon params##
        if len(phonon_params):
            if not hasattr(phonon_params[0][0],'__len__'): phonon_params=[phonon_params]*3
            Logger.raiseException('*phonon_params* should be either a phonon parameter set (list of phonon tuples) or '+\
                                  'a list of three phonon parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(phonon_params)==3), exception=ValueError)
        else: phonon_params=[[]]*3
        self.phonon_params=phonon_params
            
        ##Process lorentzian params
        if len(eps_lps):
            if not hasattr(eps_lps[0],'__len__'): eps_lps=[eps_lps]*3
            Logger.raiseException('*eps_lps* should be either a lorentzian parameter set or '+\
                                  'a list of three lorentzian parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(eps_lps)==3), exception=ValueError)
        else: eps_lps=[[]]*3
        if len(mu_lps):
            if not hasattr(mu_lps[0],'__len__'): mu_lps=[mu_lps]*3
            Logger.raiseException('*mu_lps* should be either a lorentzian parameter set or '+\
                                  'a list of three lorentzian parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(mu_lps)==3), exception=ValueError)
        else: mu_lps=[[]]*3
        self.eps_lps=eps_lps
        self.mu_lps=mu_lps
        
        ##Process voigt params##
        if len(eps_vps):
            if not hasattr(eps_vps[0],'__len__'): eps_vps=[eps_vps]*3
            Logger.raiseException('*eps_vps* should be either a Voigt parameter set or '+\
                                  'a list of three Voigt parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(eps_vps)==3), exception=ValueError)
        else: eps_vps=[[]]*3
        if len(mu_vps):
            if not hasattr(mu_vps[0],'__len__'): mu_vps=[mu_vps]*3
            Logger.raiseException('*mu_vps* should be either a Voigt parameter set or '+\
                                  'a list of three Voigt parameter sets (one for each of the '+\
                                  'principle axes, respectively).',\
                                  unless=(len(mu_vps)==3), exception=ValueError)
        else: mu_vps=[[]]*3
        self.eps_vps=eps_vps
        self.mu_vps=mu_vps
        
        BaseAnisotropicMaterial2.__init__(self,eps_infinity,mu_infinity)
    
    def invoke_property_function(self,freq,q,prop_func,props):
        
        if isinstance(prop_func,str): prop_func=getattr(self,prop_func)
        
        #props should be made into a list of property names
        if isinstance(props,str) or not hasattr(props,'__len__'): props=[props]
        else: props=list(props)
        
        #Request the corresponding attribute for each property in the prop list#
        for i,property in enumerate(props):
            if isinstance(property,str): props[i]=getattr(self,property)
            
        #Logic of the next step is as follows:
        #    Should have *props* as a list of *[3-tuple: parameter 1, 3-tuple: parameter 2, ...]*
        #    We want instead *[(a-axis parameter 1, a-axis parameter 2, ...),
        #                      (b-axis parameter 1, b-axis parameter 2, ...),
        #                      (c-axis parameter 1, c-axis parameter 2, ...)]*
        return [prop_func(freq,q,*axis_props) \
                for axis_props in zip(*props)]
        
    def epsilon_principle(self,freq,q=None):
        """Anistropic epsilon tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        eps=copy.copy(self.eps_infinity)
        for prop,prop_func in self.eps_property_functions.items():
             eps_addition=self.invoke_property_function(freq,q,prop,prop_func)
             for i in range(3):
                 eps[i]+=eps_addition[i]
        
        return anisotropic.elements_into_diag_3x3matrices(*eps)
        
    def mu_principle(self,freq,q=None):
        """Anistropic mu tensor is returned, with
        z axis corresponding to electric field polarization
        parallel to the "c" axis of the crystal."""
        
        mu=copy.copy(self.mu_infinity)
        for prop,prop_func in self.mu_property_functions.items():
             mu_addition=self.invoke_property_function(freq,q,prop,prop_func)
             for i in range(3):
                 mu[i]+=mu_addition[i]
        
        return anisotropic.elements_into_diag_3x3matrices(*mu)