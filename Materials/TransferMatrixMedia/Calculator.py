'''
Created on Apr 3, 2019

@author: Leo Lo
'''

from NearFieldOptics.Materials.material_types import *
from NearFieldOptics.Materials.TransferMatrixMedia import MatrixBuilder as mb
import sympy
import copy
import numpy as np
from common.baseclasses import ArrayWithAxes as AWA

class Calculator():
    """Calculator class calculates analytical expression and numerical value of various optical parameters.
        
    Attributes:
        The analytical expression of the following optical parameters are stored:
        - Reflection Coefficient
        - Reflectance
        - Transmission Coefficient
        - Transmittance
        - H Field
        - Reference Kernel (from Alonso-Gonzalez et al., Nature Nanotechnology 185, 2016)
        - Kernel
    """
    
    def __init__(self,transferMatrix):
        """Construct a calculator object. 
        
        Args:
            transferMatrix (TransferMatrix): a transferMatrix object constructed by the MatrixBuilder.py module, based on 
        the input material.
    
        Return:
            void
    
        """
        self.transferMatrix = transferMatrix
        self.analyticalReflectionCoefficient = None
        self.analyticalReflectance = None
        self.analyticalTransmissionCoefficient = None
        self.analyticalTransmittance = None
        self.analyticalHField = None
        self.analyticalReferenceKernel = None
        self.analyticalKernel = None
        self.numLayers = self.transferMatrix.get_layer_count()-2
    
    def assemble_analytical_reflection_coefficient(self):
        """Create an analytical expression for reflection coefficient of the entire LayeredMedia material.
        
        Args:
            None
        
        Return:
            void
        
        """
        matrix = self.transferMatrix.get_matrix()
        M11 = matrix[0,0]
        M21 = matrix[1,0]
        self.analyticalReflectionCoefficient = M21/M11
    
    def get_analytical_reflection_coefficient(self):
        """Get class variable analyticalReflectionCoefficient.
        
        Args:
             None
        
        Return:
            Analytical expression for reflection coefficient.
        
        """
        return copy.copy(self.analyticalReflectionCoefficient)
    
    def get_numerical_reflection_coefficient(self, freq, q):
        """Get numerical reflection coefficient.
            
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalReflectionCoefficient class variable. 
        Broadcast the 1D freq and q arrays into a 2D array to evaluate reflection coefficient at each combination of freq and q.
        
        Args:
            freq (array): numpy.ndarray array of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            The numerical reflection coefficient with corresponding dimension of array (based on dimension of freq and q). 
        
        """
        r = self.analyticalReflectionCoefficient
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
         
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
         
        for x in range(2, self.numLayers+2):
             
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
         
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
        
        numerics = sympy.lambdify(subs.keys(), r, modules='numpy')
        r = numerics(*subs.values())
        return r
    
    def assemble_analytical_reflectance(self):
        """Create an analytical expression for reflectance of the entire LayeredMedia material.
        
        Reflectance is the same for both p- and s-polarized lights.
        
        Args:
             None
        
        Return:
            void
        
        """
        self.analyticalReflectance = abs(self.analyticalReflectionCoefficientCoefficient)**2
    
    def get_analytical_reflectance(self):
        """Get class variable analyticalReflectance.
        
        Args:
             None
        
        Return:
            Analytical expression for reflectance.
        
        """
        return copy.copy(self.analyticalReflectance)
    
    def get_numerical_reflectance(self, freq, q):
        """Get numerical reflectance. 
             
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalReflectance class variable. 
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            The numerical reflectance with corresponding dimension of array (based on dimension of freq and q).
            
        """
        R = self.analyticalReflectance
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
        
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
        
        for x in range(2, self.numLayers+2):
            
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
        
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
        
        numerics = sympy.lambdify(subs.keys(), R, modules='numpy')
        R = numerics(*subs.values())
        return R
    
    def assemble_analytical_transmission_coefficient(self):
        """Create an analytical expression for transmission coefficient of the entire LayeredMedia material.
        
        Args:
             None
        
        Return:
            void
        
        """
        matrix = self.transferMatrix.get_matrix()
        M11 = matrix[0,0]
        self.analyticalTransmissionCoefficient = 1/M11
    
    def get_analytical_transmission_coefficient(self): 
        """Get class variable analyticalTranmissionCoefficient.
        
        Args:
             None
        
        Return:
            Analytical expression for transmission coefficient.
        
        """
        return copy.copy(self.analyticalTransmissionCoefficient)
    
    def get_numerical_transmission_coefficient(self, freq, q): 
        """Get numerical transmission coefficient. 
             
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalTransmissionCoefficient class variable. 
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            The numerical transmission coefficient with corresponding dimension of array (based on dimension of freq and q). 
        
        """
        t = self.analyticalTransmissionCoefficient
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
        
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
        
        for x in range(2, self.numLayers+2):
            
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
        
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
            
        numerics = sympy.lambdify(subs.keys(), t, modules='numpy')
        t = numerics(*subs.values())
        return t
    
    def assemble_analytical_transmittance(self):
        """Create an analytical expression for transmittance of the entire LayeredMedia material.
        
        Based on whether light is p-polarized or s-polarized (info stored in transferMatrix). 
        
        Args:
             None
        
        Return:
            void
        
        """
        epsilon_first,epsilon_last,kz_first,kz_last = sympy.symbols('epsilon_1,epsilon_{0},k_z1,k_z{0}'.format(self.numLayers+2))
        if self.transferMatrix.polarization == 'p':
            self.analyticalTransmittance = epsilon_first*kz_last/(epsilon_last*kz_first)*abs(self.analyticalTransmissionCoefficient)**2
        
        else:       #self.transferMatrix.polarization == 's':
            self.analyticalTransmittance = kz_last/kz_first*abs(self.analyticalTransmissionCoefficient)**2
            
    def get_analytical_transmittance(self):
        """Get class variable analyticalTranmittance.
        
        Args:
             None
        
        Return:
            Analytical expression for transmittance.
        
        """
        return copy.copy(self.analyticalTransmittance)
    
    def get_numerical_transmittance(self, freq, q):
        """Get numerical transmittance. 
             
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalTransmittance class variable. 
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            The numerical transmittance with corresponding dimension of array (based on dimension of freq and q). 
        
        """
        T = self.analyticalTransmittance
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
        
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
        
        for x in range(2, self.numLayers+2):
            
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
        
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
            
        numerics = sympy.lambdify(subs.keys(), T, modules='numpy')
        T = numerics(*subs.values())
        return T
    
    def assemble_analytical_H_field(self, n, side):
        """Create analytical expression of H field at either side of the n,n+1 interface; store as a class variable. 
        
        Args:
            n (int): n means that a test charge is placed at the n,n+1 interface. Each layer is indexed; 
                the entrance material has n = 1. Therefore, for a material with N layers, the index goes from 1 to N+2.
            side (str): the side of the n,n+1 interface can be either "before" or "after". The H field
                on the corresponding side is then calculated.
            
        Return:
            void
        
        """
        matrixDictionary = self.transferMatrix.matrixDictionary
        
        #check for parameter inputs
        if n > (self.numLayers+1):
            Logger.raiseException('Index exceed number of layers. n cannot be greater than {0}'.format(self.numLayers+1),exception=ValueError)
        elif n < 1:
            Logger.raiseException('Invalid index. n cannot be less than 1',exception=ValueError)
        elif side!='before' and side!='after':
            Logger.raiseException('The input to side has to either be \'before\' or \'after\'', exception=ValueError)
        
        #begin assembling matrices
        M_1_to_n = sympy.Matrix([[1,0],[0,1]])
        for x in range(2, n+1):
            M_1_to_n *= matrixDictionary["T{0}{1}".format(x-1,x)].get_matrix()
            M_1_to_n *= matrixDictionary["P{0}".format(x)].get_matrix()
            
        M_1_to_n_inv = sympy.Matrix([[M_1_to_n[1,1],-M_1_to_n[0,1]],[-M_1_to_n[1,0],M_1_to_n[0,0]]])
        
        M_n_to_end = mb.TransmissionMatrix(self.transferMatrix.polarization,n,surfaceCurrent='self').get_matrix()
        for x in range(n+1, self.numLayers+2):
            M_n_to_end *= matrixDictionary["P{0}".format(x)].get_matrix()
            M_n_to_end *= matrixDictionary["T{0}{1}".format(x,x+1)].get_matrix()
        
        beta1 = M_1_to_n_inv[0,1]
        delta1 = M_1_to_n_inv[1,1]
        alpha2 = M_n_to_end[0,0]
        gamma2 = M_n_to_end[1,0]
        
        c = sympy.symbols('c')
        
        J = mb.CurrentDensityVector().get_vector()
        inhomogeneousTerm = 4*sympy.pi/c*J/2
        b1 = 1/(beta1*gamma2-alpha2*delta1)*(gamma2*inhomogeneousTerm[0]-alpha2*inhomogeneousTerm[1])
        HfieldBefore = M_1_to_n_inv*sympy.Matrix([[0],[b1]])
        
        if side=='before':
            self.analyticalHField = HfieldBefore
            
        else: #side=='after'
#             transmission = MatrixBuilder.TransmissionMatrix(self.transferMatrix.polarization,n,surfaceCurrent='self').get_matrix()
#             transmission_inv = sympy.Matrix([[transmission[1,1],-transmission[0,1]],[-transmission[1,0],transmission[0,0]]])
#             self.analyticalHField = transmission_inv*(HfieldBefore-inhomogeneousTerm)
            M_nplus1_to_end = sympy.Matrix([[1,0],[0,1]])
            for x in range(n+1,self.numLayers+2):
                M_nplus1_to_end *= matrixDictionary["P{0}".format(x)].get_matrix()
                M_nplus1_to_end *= matrixDictionary["T{0}{1}".format(x,x+1)].get_matrix()
            a_end = 1/(beta1*gamma2-alpha2*delta1)*(delta1*inhomogeneousTerm[0]-beta1*inhomogeneousTerm[1])
            self.analyticalHField = M_nplus1_to_end*sympy.Matrix([[a_end],[0]])
    
    def get_analytical_H_field(self): 
        """Get class variable analyticalHField
        
        Args:
             None.
        
        Return:
            H field right after the (n-1,n) interface.
            
        """
        return copy.copy(self.analyticalHField)
    
    def get_numerical_H_field(self, freq, q): 
        """Get numerical H field. 
            
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalHField class variable. 
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            H field with corresponding dimension of array (based on dimension of freq and q). 
        
        """
        H = self.analyticalHField[0] + self.analyticalHField[1]
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
        subs['q'] = q
        subs['rho'] = 1
        
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
        
        for x in range(2, self.numLayers+2):
            
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
        
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
            
        numerics = sympy.lambdify(subs.keys(), H, modules='numpy')
        HFieldArray = numerics(*subs.values())
        return HFieldArray
    
    def _get_interface_position_list_(self,T):
        thickness = 0
        list = []
        num_layer = T.layerIndex-2
        for i in range(0,num_layer):
            d = T.layerDictionary['L'+str(num_layer+1-i)].get_thickness()
            thickness += d
            list = np.append(list,thickness)
        return list
    
    def get_H_field_profile(self,freq,q,start='exit',a=1.,b=1.,num_sample=100):
        """Get H field at different z positions. 
        
        If user wants to calculate H field beyond a sample, they can construct a layer of the same entrance/exit material
        with the thickness corresponding to the z position they want to calculate the H field 
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            start: the interface at which to start calculating the H field; can be either 'exit' or 'entrance'
            a (float): if start=='exit', magnitude of H field exiting material;
                       if start=='entrance', magnitude of H field entering material
            b (float): if start=='exit', magnitude of H field entering material;
                       if start=='entrance', magnitude of H field exiting material
            num_sample: number of position to sample H field
        
        """
        T = self.transferMatrix
        num_layer = T.layerIndex-2
        H_0 = np.matrix([[a],[b]])
        H_profile = []
        E_profile = []
        omega = 2*np.pi*freq
        
        if start =='exit':
            
            interface_position_list = self._get_interface_position_list_(T)
            thickness = interface_position_list[-1]
            interface_index = 0           
            next_interface_position = interface_position_list[interface_index]
            
            step_size = thickness/num_sample
            positionArray = np.linspace(0,thickness,num=int(num_sample))
            
            index = num_layer+1
            
            #Obtain the exit transmission matrix
            out_material = T.exit
            kz_out = out_material.get_kz(freq,q)
            if isinstance(kz_out,np.ndarray):
                kz_out = np.ndarray.item(out_material.get_kz(freq,q))
            epsilon_out = out_material.epsilon(freq,q)
            if isinstance(epsilon_out,np.ndarray):
                epsilon_out = np.ndarray.item(out_material.epsilon(freq,q))
            
            in_material = T.layerDictionary['L'+str(index)].get_material()
            kz_in = in_material.get_kz(freq,q)
            if isinstance(kz_in,np.ndarray):
                kz_in = np.ndarray.item(in_material.get_kz(freq,q))
            epsilon_in = in_material.epsilon(freq,q)
            if isinstance(epsilon_in,np.ndarray):
                epsilon_in = np.ndarray.item(in_material.epsilon(freq,q))
            last_interface = T.layerDictionary['S'+str(index)+str(index+1)]
            sigma = last_interface.conductivity(freq)
            
            #testing
            print('epsilon out type: ' + str(type(epsilon_out)))
            print('epsilon in type: ' + str(type(epsilon_in)))
            print('kz in type: ' + str(type(kz_in)))
            print('kz out type: ' + str(type(kz_out)))
            print('sigma type: ' + str(type(sigma)))
            print('omega type: ' + str(type(omega)))
            
            
            eta = epsilon_in*kz_out/(epsilon_out*kz_in)
            xi = 4*np.pi*(sigma/29979245368)*kz_out/(epsilon_out*omega)
            
            #testing
            print('xi type: ' + str(type(xi)))
            print('eta type: ' + str(type(eta)))
            
            exit_transmission_matrix = 1/2*np.matrix([[1+eta+xi,1-eta-xi],
                                                      [1-eta+xi,1+eta-xi]])
            H_current = exit_transmission_matrix*H_0
            H_profile = np.append(H_profile,H_current.sum())
            E_current = (H_current[0]-H_current[1])*29979245368*kz_in/(omega*epsilon_in*q)
            E_profile = np.append(E_profile,E_current)
            
            for z in positionArray[1:]:
                
                if z > next_interface_position:
                    #complete distance in out_material
                    d = step_size - (z-next_interface_position)
                    propagation_matrix = np.matrix([[np.exp(-1j*kz_in*d) , 0],
                                                [0 , np.exp(1j*kz_in*d)]])
                    H_current = propagation_matrix*H_current
                    
                    #obtain the H field across the interface
                    index = index-1
                    out_material = in_material
                    kz_out = kz_in
                    epsilon_out = epsilon_in
                    
                    in_material = T.layerDictionary['L'+str(index)].get_material()
                    kz_in = in_material.get_kz(freq,q)
                    if isinstance(kz_in,np.ndarray):
                        kz_in = np.ndarray.item(in_material.get_kz(freq,q))
                    epsilon_in = in_material.epsilon(freq,q)
                    if isinstance(epsilon_in,np.ndarray):
                        epsilon_in = np.ndarray.item(in_material.epsilon(freq,q))
                    last_interface = T.layerDictionary['S'+str(index)+str(index+1)]
                    sigma = last_interface.conductivity(freq)
                
                    eta = epsilon_in*kz_out/(epsilon_out*kz_in)
                    xi = 4*np.pi*(sigma/29979245368)*kz_out/(epsilon_out*omega)
                    
                    transmission_matrix = 1/2*np.matrix([[1+eta+xi,1-eta-xi],
                                                         [1-eta+xi,1+eta-xi]])
                    H_current = transmission_matrix*H_current
                    
                    #complete remaining distance in in_material
                    remaining_d = z-next_interface_position
                    propagation_matrix = np.matrix([[np.exp(-1j*kz_in*d) , 0],
                                                [0 , np.exp(1j*kz_in*d)]])
                    H_current = propagation_matrix*H_current
                    H_profile = np.append(H_profile,H_current.sum())
                    E_current = (H_current[0]-H_current[1])*29979245368*kz_in/(omega*epsilon_in*q)
                    E_profile = np.append(E_profile,E_current)
                    
                    interface_index += 1         
                    next_interface_position = interface_position_list[interface_index]
                
                elif z == next_interface_position:
                    propagation_matrix = np.matrix([[np.exp(-1j*kz_in*step_size) , 0],
                                                [0 , np.exp(1j*kz_in*step_size)]])
                
                    H_current = propagation_matrix*H_current
                    H_profile = np.append(H_profile,H_current.sum())
                    E_current = (H_current[0]-H_current[1])*29979245368*kz_in/(omega*epsilon_in*q)
                    E_profile = np.append(E_profile,E_current)
                    
                    if z!=thickness:
                        #obtain the H field across the interface
                        index = index-1
                        out_material = in_material
                        kz_out = kz_in
                        epsilon_out = epsilon_in
                        
                        in_material = T.layerDictionary['L'+str(index)].get_material()
                        kz_in = in_material.get_kz(freq,q)
                        if isinstance(kz_in,np.ndarray):
                            kz_in = np.ndarray.item(in_material.get_kz(freq,q))
                        epsilon_in = in_material.epsilon(freq,q)
                        if isinstance(epsilon_in,np.ndarray):
                            epsilon_in = np.ndarray.item(in_material.epsilon(freq,q))
                        last_interface = T.layerDictionary['S'+str(index)+str(index+1)]
                        sigma = last_interface.conductivity(freq)
                    
                        eta = epsilon_in*kz_out/(epsilon_out*kz_in)
                        xi = 4*np.pi*(sigma/29979245368)*kz_out/(epsilon_out*omega)
                        
                        transmission_matrix = 1/2*np.matrix([[1+eta+xi,1-eta-xi],
                                                             [1-eta+xi,1+eta-xi]])
                        H_current = transmission_matrix*H_current
                
                else:
                    propagation_matrix = np.matrix([[np.exp(-1j*kz_in*step_size) , 0],
                                                [0 , np.exp(1j*kz_in*step_size)]])
                
                    H_current = propagation_matrix*H_current
                    H_profile = np.append(H_profile,H_current.sum())
                    E_current = (H_current[0]-H_current[1])*29979245368*kz_in/(omega*epsilon_in*q)
                    E_profile = np.append(E_profile,E_current)
                    
            r = np.linspace(0,thickness,num=num_sample)
            return AWA(E_profile,axes=[r*1e7],axis_names=['distance from exit (nm)']),\
                   AWA(H_profile,axes=[r*1e7],axis_names=['distance from exit (nm)'])
            
        elif start == 'entrance':
            
            return AWA(H_profile,axes=[r*1e7],axis_names=['distance from entrance (nm)'])
        
        else:
            Logger.raiseException('Invalid input for start. Only accept \'exit\' or \'entrance\'', exception=ValueError)            
        
    def assemble_analytical_reference_kernel(self):
        """Create an analytical expression for Coulomb kernel from Alonso-Gonzalez et al., Nature Nanotechnology 185, 2016.
        
        The material is a graphene sheet encapsulated by two uniaxial layers in an isotropic medium (permittivity of 
        medium above and below the material can be different). 
        
        Args:
             None
        
        Return:
            void
        
        """
        epsilon_x,epsilon_z,epsilon_a,epsilon_b,e,q,d1,d2 = sympy.symbols('epsilon_x,epsilon_z,epsilon_a,epsilon_b,e,q,d1,d2')
        v_q = 4*sympy.pi*e**2/(q*(epsilon_a+epsilon_b))
        epsilon_tilta = (epsilon_a*epsilon_b+epsilon_x*epsilon_z)/(epsilon_a+epsilon_b)
        V = v_q*sympy.Rational(1,2)*(sympy.sqrt(epsilon_x*epsilon_z)
                                     + (epsilon_a+epsilon_b)*sympy.tanh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d1+d2))
                                     + (epsilon_b-epsilon_a)*sympy.sinh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d1-d2))/sympy.cosh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d1+d2))
                                     + (sympy.sqrt(epsilon_x*epsilon_z)-epsilon_a*epsilon_b/sympy.sqrt(epsilon_x*epsilon_z))*sympy.cosh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d2-d1))/sympy.cosh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d1+d2))
                                     + epsilon_a*epsilon_b/sympy.sqrt(epsilon_x*epsilon_z)
                                     )/(
                                         sympy.sqrt(epsilon_x*epsilon_z)+epsilon_tilta*sympy.tanh(q*sympy.sqrt(epsilon_x/epsilon_z)*(d1+d2))
                                         )
        
        self.analyticalReferenceKernel = V
    
    def get_analytical_reference_kernel(self): 
        """Get analytical Coulomb kernel from Alonso-Gonzalez et al., Nature Nanotechnology 185, 2016.
        
        The material is a graphene sheet encapsulated by two uniaxial layers in an isotropic medium (permittivity of 
        medium above and below the material can be different). 
            
        Args:
            None.
        
        Return:
            Analytical expression of Coulomb kernel for an graphene encapsulated by two uniaxial materials
            in an isotropic medium.
        
        """
        return copy.copy(self.analyticalReferenceKernel)
    
    def get_numerical_reference_kernel(self,freq,q,material,d1,d2,epsilon_a=1,epsilon_b=1):
        """Get numerical Coulomb kernel from Alonso-Gonzalez et al., Nature Nanotechnology 185, 2016.
            The material is a graphene sheet encapsulated by two uniaxial layers in an isotropic medium.
        
        Args:
            q (float array): an array of in-plane momenta of incident light.
            epsilon_x (float): the complex in-plane relative permittivity of uniaxial material
            epsilon_z (float): the complex out-of-plane relative permittivity of uniaxial material
            epsilon_a (float): the complex relative permittivity of isotropic medium above the sample 
            epsilon_b (float): the complex relative permittivity of isotropic medium below the sample
        
        Return:
            An array of numerical value of Coulomb kernel (as a function of q) for an graphene encapsulated by two 
            uniaxial materials in an isotropic medium.
        
        """
        V = self.analyticalReferenceKernel
        
        subs = {}
        subs['epsilon_a'] = epsilon_a
        subs['epsilon_b'] = epsilon_b
        subs['e'] = 1        #"normalized"
        subs['d1'] = d1
        subs['d2'] = d2
        subs['q'] = q
        
        if (type(material)==BaseIsotropicMaterial or type(material)==IsotropicMaterial):
            subs[sympy.symbols('epsilon_x')] = material.epsilon(freq,q)
            subs[sympy.symbols('epsilon_z')] = material.epsilon(freq,q)
        elif (type(material)==BaseAnisotropicMaterial or type(material)==AnisotropicMaterial):
            subs[sympy.symbols('epsilon_x')] = material.ordinary_epsilon(freq,q)
            subs[sympy.symbols('epsilon_z')] = material.extraordinary_epsilon(freq,q)
        else:
            Logger.raiseException('Invalid material. Accept only material of type BaseIsotropicMaterial,\
            IsotropicMaterial,BaseAnisotropicMaterial, or AnisotropicMaterial.',exception=ValueError)
        
        numerics = sympy.lambdify(subs.keys(), V, modules='numpy')
        potentialArray = numerics(*subs.values())
          
        return potentialArray
    
    def assemble_analytical_reference_kernel_2(self):
        epsilon_x,epsilon_z,epsilon_a,epsilon_b,e,q,d = sympy.symbols('epsilon_x,epsilon_z,epsilon_a,epsilon_b,e,q,d')
        v_q = 4*sympy.pi*e**2/(q*(epsilon_a+epsilon_b))
        V = v_q*sympy.Rational(1,2)*(
                sympy.sqrt(epsilon_x*epsilon_z)+(epsilon_a+epsilon_b)*sympy.tanh(q*d*sympy.sqrt(epsilon_x/epsilon_z))
            )/(
                sympy.sqrt(epsilon_x*epsilon_z)+(epsilon_x*epsilon_z+epsilon_b*epsilon_a)*sympy.tanh(q*d*sympy.sqrt(epsilon_x/epsilon_z))/(epsilon_a+epsilon_b)
                )
        
        self.analyticalReferenceKernel = V
            
    def get_numerical_reference_kernel_2(self,freq,q,material,d,epsilon_a=1,epsilon_b=1):
        
        V = self.analyticalReferenceKernel
        
        subs = {}
        subs['epsilon_a'] = epsilon_a
        subs['epsilon_b'] = epsilon_b
        subs['e'] = 1        #"normalized"
        subs['d'] = d
        subs['q'] = q
        
        if (type(material)==BaseIsotropicMaterial or type(material)==IsotropicMaterial):
            subs['epsilon_x'] = material.epsilon(freq,q)
            subs['epsilon_z'] = material.epsilon(freq,q)
        elif (type(material)==BaseAnisotropicMaterial or type(material)==AnisotropicMaterial):
            subs['epsilon_x'] = material.ordinary_epsilon(freq,q)
            subs['epsilon_z'] = material.extraordinary_epsilon(freq,q)
        else:
            Logger.raiseException('Invalid material. Accept only material of type BaseIsotropicMaterial,\
            IsotropicMaterial,BaseAnisotropicMaterial, or AnisotropicMaterial.',exception=ValueError)
        
        numerics = sympy.lambdify(subs.keys(), V, modules='numpy')
        potentialArray = numerics(*subs.values())
          
        return potentialArray
    
    def direct_numerical_reference_kernel_2(self,freq,q,material,d,epsilon_a=1,epsilon_b=1):
        
        if (type(material)==BaseIsotropicMaterial or type(material)==IsotropicMaterial):
            epsilon_x = material.epsilon(freq,q)
            epsilon_z = material.epsilon(freq,q)
        elif (type(material)==BaseAnisotropicMaterial or type(material)==AnisotropicMaterial):
            epsilon_x = material.ordinary_epsilon(freq,q)
            epsilon_z = material.extraordinary_epsilon(freq,q)
        else:
            Logger.raiseException('Invalid material. Accept only material of type BaseIsotropicMaterial,\
            IsotropicMaterial,BaseAnisotropicMaterial, or AnisotropicMaterial.',exception=ValueError)
        
        e = 1
        v_q = -4*np.pi*e/(q*(epsilon_a+epsilon_b))
        V = v_q*(
                safe_sqrt(epsilon_x*epsilon_z)+epsilon_b*np.tanh(q*d*safe_sqrt(epsilon_x/epsilon_z))
            )/(
                safe_sqrt(epsilon_x*epsilon_z)+(epsilon_x*epsilon_z+epsilon_b*epsilon_a)*np.tanh(q*d*safe_sqrt(epsilon_x/epsilon_z))/(epsilon_a+epsilon_b)
                )
        return V
    
    
    def assemble_analytical_kernel(self,n,side):
        """Create analytical expression of Coulomb kernel from transfer matrix method.
        
        Position of the kernel is at either side of the n,n+1 interface.
        Analytical kernel is stored as a class variable.
        
        Args:
            n (int): n means that a test charge is placed at the n,n+1 interface. Each layer is indexed; 
                the entrance material has n = 1. Therefore, for a material with N layers, the index goes from 1 to N+2.
            side (str): the side of the n,n+1 interface can be either "before" or "after". The H field
                on the corresponding side is then calculated.
        
        Return:
            void
        
        """
        if side == 'before':
            epsilon_n = sympy.symbols('epsilon_{}'.format(n))
            k_n = sympy.symbols('k_z{}'.format(n))
        
        elif side == 'after':
            epsilon_n = sympy.symbols('epsilon_{}'.format(n+1))
            k_n = sympy.symbols('k_z{}'.format(n+1))
            
        else:
            Logger.raiseException('The input to side has to either be \'before\' or \'after\'', exception=ValueError)
        
        self.assemble_analytical_H_field(n,side)
        omega,c,q = sympy.symbols('omega,c,q')
        a = self.get_analytical_H_field()[0]
        b = self.get_analytical_H_field()[1]
        
        #2020.05.07 ASM:  Removed an overall minus sign which conversed the correct free-space answer for air/air stack
        self.analyticalKernel = sympy.I*c*k_n/(omega*epsilon_n*q)*(b-a)
    
    def get_analytical_kernel(self): 
        """Get analytical Coulomb kernel from transfer matrix method.
        
        Args:
             None
        
        Return:
            analytical expression of the Coulomb kernel (self.analyticalKernel) at the n-1,n interface.
            
        """
        return copy.copy(self.analyticalKernel)
    
    def get_numerical_kernel(self, freq, q):
        """Get numerical Coulomb kernel from transfer matrix method.
            
        Use lambdify function to substitute numerical values into analytical expression stored in 
        self.analyticalHField class variable.
        
        Args:
            freq (array): numpy.ndarray of frequencies of incident light; in unit of cm^-1
            q (array): numpy.ndarray of in-plane momenta associated with incident light
        
        Return:
            The numerical Coulomb kernel with corresponding dimension of array (based on dimension of freq and q).
        
        """

        V = self.analyticalKernel
        
        entranceMaterial = self.transferMatrix.entrance
        exitMaterial = self.transferMatrix.exit
        layerDictionary = self.transferMatrix.layerDictionary
        
        subs = {}
        subs['c'] = 3e10
        subs['omega'] = 2*np.pi*freq
        subs['q'] = q
        subs['rho'] = 1
        
        #for first boundary
        subs['k_z1'] = entranceMaterial.get_kz(freq,q)
        subs['epsilon_1'] = entranceMaterial.epsilon(freq,q)
        subs['mu_1'] = entranceMaterial.mu(freq,q)
         
        for x in range(2, self.numLayers+2):
             
            layer = layerDictionary['L'+str(x)]
            material = layer.get_material()
            surface = layerDictionary['S'+str(x-1)+str(x)]
            subs['k_z{}'.format(x)] = material.get_kz(freq,q)
            subs['z{}'.format(x)] = layer.get_thickness()
            subs['sigma{0}{1}'.format(x-1,x)] = surface.conductivity(freq)
            subs['epsilon_{}'.format(x)] = material.epsilon(freq,q)
            subs['mu_{}'.format(x)] = material.mu(freq,q)
         
        #for last boundary
        subs['k_z{}'.format(self.numLayers+2)] = exitMaterial.get_kz(freq,q)
        subs['epsilon_{}'.format(self.numLayers+2)] = exitMaterial.epsilon(freq,q)
        subs['mu_{}'.format(self.numLayers+2)] = exitMaterial.mu(freq,q)
        surface = layerDictionary['S'+str(self.numLayers+1)+str(self.numLayers+2)]
        subs['sigma{0}{1}'.format(self.numLayers+1,self.numLayers+2)] = surface.conductivity(freq)
         
        numerics = sympy.lambdify(subs.keys(), V, modules='numpy')
        potentialArray = numerics(*subs.values())
                
        return potentialArray
        
