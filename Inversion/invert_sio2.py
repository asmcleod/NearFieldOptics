import os
import numpy
from common.misc import extract_array
from matplotlib import pyplot
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat
from NearFieldOptics.Inversion import inversion_wrapper as inv

root_dir=os.path.dirname(__file__)


tip.LRM.geometric_params['geometry']='cone' #shape of probe, could also be "cone"
tip.LRM.geometric_params['a']=30 #tip radius, in nm
#This uses charge distributions calculated at 1000 cm-1, capturing proper retardation effects in the mid-IR.
#Could also be =0 for a quasi-static approximation.  You'll find it doesn't make much difference which you choose for SiO2.
tip.LRM.load_params['freq']=1000*30e-7

##Define a priori parameters of our thin film###
ReferenceMaterial=mat.Si #data was normalized to silicon (known optical constants)
RefSignal=None
layer_thickness=300 #in nanometers
SampleMaterial=mat.IsotropicMaterial(eps_infinity=1) #Bogus sample material
Film=mat.LayeredMedia((SampleMaterial,layer_thickness*1e-7),exit=mat.Si) #layer representing our sample
enforce_physicality=True #Force the sample material to be physical

def SignalFromThinFilm(freq,x,y,tip_model=tip.LRM,**kwargs):
    
    harmonic = kwargs['harmonic']
    global Layer,ReferenceMaterial,\
           RefSignal,enforce_physicality,Film
    
    #Assign sample material (top-most layer material) the epsilon value
    if enforce_physicality and y<0: y=numpy.abs(y) #Force epsilon to be in the upper half plane.
    Film.get_layers()[0].get_material().eps_infinity=x+1j*y
    
    try:
        if RefSignal is None: raise NameError
    except NameError:
        RefSignal=tip_model(freq,rp=ReferenceMaterial.reflection_p,**kwargs)
    
    #Compute signal using built-in reflection coefficient of the film object#
    SampleSignal=tip_model(freq,rp=Film.reflection_p,**kwargs)
    
    norm_val=SampleSignal['signal_%i'%harmonic]\
            /RefSignal['signal_%i'%harmonic]
            
    #Don't want to the model to reload its charge data from file each time, cache it instead#
    if tip_model is tip.LRM:
        tip.LRM.load_params['reload_model']=False
        tip.verbose=False
        
    #Take value at first frequency (there's only one frequency anyway)
    if norm_val.ndim: return norm_val[0]
    else: return norm_val

##Settings for the inversion##
#jacobian_thresh = local determinant of the "NF map" jacobian below which the fitting routine should go into "free fall recovery" mode
#omega_coeff = should be called "period_coeff", the period of oscillation of the oscillator equals h*omega_coeff, with h the local spacing in frequency
#freq_reverse = True to perform fitting in reverse, useful in cases where forward fitting was crummy
#data_threshold = threshold difference between data and model above which fitting goes into "convergence mode"
#iteration_max = maximum number of iterations to attempt in "convergence mode"
#additional arguments (harmonic, amplitude, Nzs, etc.) are routed to the signal model, in this case the lightning rod model
inversion_kwargs={"jacobian_thres":0.0, "omega_coeff": 6,'freq_reverse':False,
                  'iteration_max':5,'data_threshold':.1,
                  'amplitude':80,'Nzs':15}

##Shall we now invert S3 data for a 300nm SiO2 film on silicon?##
def InvertSiO2(harmonic=3):
    
    global root_dir
    data_path=os.path.join(root_dir,'InversionData',\
                           'SiO2_S%i_spectrum_FromTrenchLinescan_PhaseCorrected.csv'%\
                           harmonic)
    freqs,s2r,s2i=extract_array(open(data_path)).transpose().astype(float)
                    
    #Initial conditions (guess value) for starting epsilon#
    eps0=1; deps0dw=0 #Who can guess what the initial slope should be? Put zero.
    
    #Run the inversion#
    global inversion_kwargs
    global best_fit #Just in case something crashes subsequent to inversion, we have a global handle to our result
    
    best_fit=inv([eps0.real,eps0.imag,\
                  deps0dw.real,deps0dw.imag],\
                  s2r,s2i,freqs_D=freqs,SModel=SignalFromThinFilm,\
                  freqs=freqs,harmonic=harmonic,**inversion_kwargs)
    
    #Plot the result
    pyplot.figure()
    pyplot.subplot(121)
    pyplot.title('Signal Data / Model Comparison',fontsize=22)
    pyplot.plot(freqs,numpy.sqrt(s2i**2+s2r**2),marker='o',ls='',label='Data')
    numpy.abs(best_fit['sig_bf_AWA']).plot(lw=2,label='Model')
    pyplot.xlabel(r'$\omega\,[cm^{-1}]$',fontsize=22)
    pyplot.ylabel(r'$S_%i$'%harmonic,fontsize=25)
    pyplot.legend(loc='best',fancybox=True,shadow=True)
    
    pyplot.subplot(122)
    pyplot.title('Extracted Optical Constants',fontsize=22)
    best_fit['beta_AWA'].real.plot(color='r',label=r'$\epsilon_1$')
    best_fit['beta_AWA'].imag.plot(color='b',label=r'$\epsilon_2$')
    pyplot.xlabel(r'$\omega\,[cm^{-1}]$',fontsize=22)
    pyplot.ylabel(r'$\epsilon$',fontsize=25)
    pyplot.legend(loc='best',fancybox=True,shadow=True)
    
    pyplot.tight_layout()
    
    return best_fit
