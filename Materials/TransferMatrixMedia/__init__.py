import numpy; np=numpy
from common import misc
from NearFieldOptics.Materials import Air,LayeredMedia
from NearFieldOptics.Materials.material_types import *
from NearFieldOptics.Materials.material_types import _prepare_freq_and_q_holder_
from NearFieldOptics.Materials.TransferMatrixMedia import MatrixBuilder as mb
from NearFieldOptics.Materials.TransferMatrixMedia import Calculator
# import MatrixBuilder as mb
# import Calculator

class LayeredMediaTM(LayeredMedia):
    
    def __init__(self,*layers,layerArrayGUIInput=None,**kwargs):
         
        if layerArrayGUIInput==None:
            self.set_layers(*layers)
         
        else:
            print(layerArrayGUIInput!= None)
            print(layerArrayGUIInput)
            self.set_layers(*layerArrayGUIInput)
        
        #Set default entrance/exit materials
        exkwargs=misc.extract_kwargs(kwargs,entrance=Air,exit=Air)
        self.set_entrance(exkwargs['entrance'])
        self.set_exit(exkwargs['exit'])
         
        self.T_p = mb.TransferMatrix(self,polarization='p')
        self.T_s = mb.TransferMatrix(self,polarization='s')
        
    def reflection_s(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Get numerical reflection coefficient for s-polarized light.
        
        First the analytical expression for reflection coefficient is assembled. 
        Then the numerical values are evaluated in the helper class Calculator. 
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light; in unit of cm^-1
        
        Return:
            Numerical reflection coefficient with corresponding dimension of 
            array (based on dimension of freq and q).
        
        """
        freq,q,rsAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_s)
        C.assemble_analytical_reflection_coefficient()
        rs = C.get_numerical_reflection_coefficient(freq,q)
        rsAWA+=rs
        return rsAWA
        
    def analytical_reflection_s(self):
        """Get sympy analytical expression of reflection coefficient for s-polarized light."""
        C = Calculator.Calculator(self.T_s)
        C.assemble_analytical_reflection_coefficient()
        rs = C.get_analytical_reflection_coefficient()
        return rs
        
    def reflection_p(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Get numerical reflection coefficient for p-polarized light.
        
        First the analytical expression for reflection coefficient is assembled. 
        Then the numerical values are evaluated in the helper class Calculator. 
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light; in unit of cm^-1
        
        Return:
            The numerical reflection coefficient with corresponding dimension of 
            array (based on dimension of freq and q).
        
        """        
        freq,q,rpAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_reflection_coefficient()
        rp = C.get_numerical_reflection_coefficient(freq,q)
        rpAWA+=rp
        return rpAWA
    
    def analytical_reflection_p(self):
        """Get sympy analytical expression of reflection coefficient for p-polarized light."""
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_reflection_coefficient()
        rp = C.get_analytical_reflection_coefficient()
        return rp
    
    def coulomb_kernel(self,freq,q=0,layer_number=2,mode='after',\
                       angle=None,entrance=None,exit=None,**kwargs):
        """Get numerical reflection coefficient for p-polarized light.
        
        First the analytical expression for reflection coefficient is assembled. 
        Then the numerical values are evaluated in the helper class Calculator. 
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light; in unit of cm^-1
        
        Return:
            The numerical reflection coefficient with corresponding dimension of 
            array (based on dimension of freq and q).
        
        """        
        freq,q,K_AWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_p)
        
        C.assemble_analytical_kernel(layer_number,mode)
        K=C.get_numerical_kernel(freq,q)
        
        K_AWA+=K
        return K_AWA
    
    def analytical_coulomb_kernel(self,layer_number=2,mode='after'):
        """Get sympy analytical expression of reflection coefficient for p-polarized light."""
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_reflection_coefficient()
        rp = C.get_analytical_coulomb_kernel(layer_number,mode)
        return rp
    
    def transmission_s(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Get numerical transmission coefficient for s-polarized light.
        
        First the analytical expression for transmission coefficient is assembled. 
        Then the numerical values are evaluated in the helper class Calculator. 
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light; in unit of cm^-1
        
        Return:
            Numerical transmission coefficient with corresponding dimension of 
            array (based on dimension of freq and q).
        
        """
        freq,q,tsAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_s)
        C.assemble_analytical_transmission_coefficient()
        ts = C.get_numerical_transmission_coefficient(freq,q)
        tsAWA+=ts
        return tsAWA
    
    def analytical_transmission_s(self):
        """Get sympy analytical expression of transmission coefficient for s-polarized light."""
        C = Calculator.Calculator(self.T_s)
        C.assemble_analytical_transmission_coefficient()
        ts = C.get_analytical_transmission_coefficient()
        return ts
        
    def transmission_p(self,freq,q=0,angle=None,\
                     entrance=None,exit=None,**kwargs):
        """Get numerical transmission coefficient for p-polarized light.
        
        First the analytical expression for transmission coefficient is assembled. 
        Then the numerical values are evaluated in the helper class Calculator. 
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light; in unit of cm^-1
        
        Return:
            The numerical transmission coefficient with corresponding dimension of 
            array (based on dimension of freq and q).
        
        """
        freq,q,tpAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_transmission_coefficient()
        tp = C.get_numerical_transmission_coefficient(freq,q)
        tpAWA+=tp
        return tpAWA
    
    def analytical_transmission_p(self):
        """Get sympy analytical expression of transmission coefficient for p-polarized light."""
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_transmission_coefficient()
        tp = C.get_analytical_transmission_coefficient()
        return tp
    
    def h_field(self,freq,q=0,index=1,angle=None,\
                     entrance=None,exit=None,**kwargs):
        freq,q,hAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_H_field(index,'before')
        h = C.get_numerical_H_field(freq,q)
        hAWA+=h
        return hAWA
    
    def analytical_h_field(self,index,side):
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_H_field(index,side)
        h = C.get_analytical_H_field()
        return h
        
    def Coulomb_kernel(self,freq,q=0,index=1,angle=None,\
                     entrance=None,exit=None,**kwargs):
        freq,q,kAWA = _prepare_freq_and_q_holder_(freq,q,angle=angle,entrance=entrance)
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_kernel(index,'before')
        k = C.get_numerical_kernel(freq,q)
        kAWA+=k
        return kAWA
        
    def analytical_Coulomb_kernel(self,index,side):
        
        C = Calculator.Calculator(self.T_p)
        C.assemble_analytical_kernel(index,side)
        k = C.get_analytical_kernel()
        return k