import os
from datetime import datetime
from numpy import *
import cPickle
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat
from common.misc import extract_array
from common.numerics import Spectrum
from matplotlib.pyplot import *
from NearFieldOptics import Inversion as inv

RefSignal=None

def Olivine_SModel(freq,x,y,tip_model=tip.LRM,**kwargs):
    """Inverting data from a material in the frequency range ~ 1000 cm-1"""
    
    global RefSignal,SampleSignal
    
    tip.LRM.load_params['freq']=1000*30/1e7
    tip.verbose=False
    harmonic = kwargs["harmonic"]
    RefMaterial=mat.Au
    if RefSignal is None:
        RefSignal=tip_model(freq,rp=RefMaterial.reflection_p,**kwargs)
    #print type(RefSignal)
        
    SampleMaterial=mat.IsotropicMaterial(eps_infinity=x+1j*y)
    SampleSignal=tip_model(freq,rp=SampleMaterial.reflection_p,**kwargs)
    
    if tip_model is tip.LRM: tip.LRM.load_params['reload_model']=False
    
    norm_val=SampleSignal['signal_%i'%harmonic]\
            /RefSignal['signal_%i'%harmonic]
    
    return norm_val.flatten()[0]

def SiC_SModel(freq,x,y,tip_model=tip.LRM,**kwargs):
    """Inverting data from a material in the frequency range ~ 900 cm-1"""
    
    global RefSignal,SampleSignal
    
    tip.LRM.load_params['freq']=900*30/1e7
    tip.verbose=False
    RefSignal = None
    harmonic = kwargs["harmonic"]
    RefMaterial=mat.Au
    SampleMaterial=mat.IsotropicMaterial(eps_infinity=1)
    if RefSignal is None:
        RefSignal=tip_model(freq,rp=RefMaterial.reflection_p,**kwargs)
        
    SampleMaterial.eps_infinity=x+1j*y
    SampleSignal=tip_model(freq,rp=SampleMaterial.reflection_p,**kwargs)
    
    if tip_model is tip.LRM: tip.LRM.load_params['reload_model']=False
    
    norm_val=SampleSignal['signal_%i'%harmonic]\
            /RefSignal['signal_%i'%harmonic]
    
    return norm_val[0]

def SiO2_SModel(freq,x,y,tip_model=tip.LRM,**kwargs):
    """Inverting data from a material with thickness"""
    
    tip.LRM.load_params['freq']=1000*30/1e7
    tip.verbose=False
    global RefSignal
    RefSignal=None

    RefMaterial=mat.Si
    SampleMaterial=mat.IsotropicMaterial(eps_infinity=1)
    thickness_nm=300
    SampleLayers=mat.LayeredMedia((SampleMaterial,thickness_nm*1e-7),exit=mat.Si)
    harmonic = kwargs['harmonic']
    
    if RefSignal is None:
        RefSignal=tip_model(freq,rp=RefMaterial.reflection_p,**kwargs)
        
    SampleMaterial.eps_infinity=x+1j*y
    SampleSignal=tip_model(freq,rp=SampleLayers.reflection_p,**kwargs)
    
    if tip_model is tip.LRM: tip.LRM.load_params['reload_model']=False
    
    norm_val=SampleSignal['signal_%i'%harmonic]\
            /RefSignal['signal_%i'%harmonic]
    
    return norm_val[0]

###############################
##INVERT GENERIC OBJECT CODE ##
###############################

def invert_object(SModel, file_type = 'pickle',eps_0= .1+0j,**kwargs):
    """Invert data read from a file"""
    
    input_freqs = kwargs['input_freqs']
    global spectrum,Beta_inics,Sig_reals,Sig_imags,freqs_D
    data_dir = kwargs['data_dir']
    data_filename = kwargs['data_filename']

    ############################FILE TYPES####################################
    if (file_type == 'pickle'):
        dic = cPickle.load(open(os.path.join(data_dir,data_filename)))
        
        if 'normalized_complex_spectrum' not in dic:
            signal = dic['s3']
            phase = dic['p3']
            spectrum = signal*exp(1j*phase)
            freqs_D = signal.axes[0]
            Sig_reals = spectrum.real
            Sig_imags = spectrum.imag
            #figure()
            #title("Input data")
            #plot(freqs_D, Sig_imags, label = 'sig_imag', color = 'r')
            #plot(freqs_D, Sig_reals, label = 'sig_real', color = 'b')
            #legend()
            #show()
        else:
            spectrum = dic['normalized_complex_spectrum']
            freqs_D = spectrum.axes[0]
            Sig_reals = spectrum.real
            Sig_imags = spectrum.imag
    
    elif (file_type == 'csv'):
        freqs_D,Sig_reals,Sig_imags=extract_array(open(os.path.join(data_dir,data_filename))).astype(float).T
        
    else:
        print "No method available for your file type"
    ############################FREQ MANIPULATION##################
    if (input_freqs == None):
        freqs = freqs_D
    else:
        freqs = input_freqs
        
    ###################################
    ##Assemble arguments to inversion##
    ###################################
            
    Beta_inics = [eps_0.real,eps_0.imag,0.,0.]
    print Beta_inics
    best_fit = inv.inversion_wrapper(Beta_inics,Sig_reals,Sig_imags,freqs_D, SModel, freqs= freqs,**kwargs)
    return best_fit
