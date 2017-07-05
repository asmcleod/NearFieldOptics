from numpy import *
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat
from scipy.interpolate import interp1d
from common.misc import extract_array
from common.numerics import Spectrum
from matplotlib.pyplot import *
from common.plotting import *
from NearFieldOptics import Inversion as inv

kwargs={"jacobian_thres":0.0, "omega_coeff": 12,'freq_reverse':False,\
        'iteration_max':5,'data_threshold':1}

def SModel(w,x,y,**kwargs):
    """A (nearly) analytic function in external inputs x,y (and 'frequency' value w) and spits out a complex number"""
    
    return x**2+y**2*1j#(cos(x)+sin(y)) + 1j*(cos(y)+sin(x))

def TestInversion(wsteps=1000,SModel=SModel,**kwargs):
    
    w0 = 0 # starting frequency
    wt = 1 # ending frequency
    #wsteps steps in between frequencies
    freqs_D=linspace(w0,wt,wsteps)
    rad = 3  # radius of the circle for comparison path
    freqs=linspace(w0,wt,500)
    
    "Sd1"
    xs = rad * cos(2 * pi * (freqs_D-w0) / float(wt-w0)) +2*rad
    ys = rad * sin(2 * pi * (freqs_D-w0) / float(wt-w0)) + (2)*rad 
    
    SModelInterp = interp1d(freqs_D,SModel(freqs,xs,ys))
    Sd = SModelInterp(freqs_D)
    Sd1_reals = Sd.real
    Sd1_imags = Sd.imag
    
    inics = [xs[0],ys[0],0,0]
    
    Sig_reals = Sd1_reals+4*random.randn(wsteps)
    Sig_imags = Sd1_imags+4*random.randn(wsteps)
    
    best_fit = inv.inversion_wrapper(inics,Sig_reals,Sig_imags,freqs_D, SModel,**kwargs)
    
    Sdata_interp = interp1d(freqs_D,(Sig_reals + 1j*Sig_imags))
    
    best_fit_signal = best_fit['sig_bf_AWA']
    
    figure();subplot(121)
    plot(freqs_D, abs(best_fit_signal),'--', label = 'Best fit Signal', linewidth = 2.5, color = "g")
    plot(freqs, abs(Sdata_interp(freqs)), label = 'Signal Data' , color = 'k')
    xlabel("Frequency")
    ylabel("Arbitrary Units")
    title("Comparing Signal Data to Best Fit")
    legend(loc='best')
    show()
    
    subplot(122)
    plot(xs,ys,label='True parameter values',color='k',marker='o')
    plot(best_fit['beta_AWA'].real,best_fit['beta_AWA'].imag,label='Best fit parameter values',color='g',lw=2.5,ls='--')
    xlabel("Parameter X")
    ylabel("Parameter Y")
    title("Comparing Parameter Values to Best Fit")
    legend(loc='best'); grid()
    show()