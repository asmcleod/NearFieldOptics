'''
Created on Apr 3, 2019

@author: Leo Lo
'''

from NearFieldOptics.Materials import *
import copy
import sympy

class TransferMatrix:
    """TransferMatrix class assembles and stores the transfer matrix based on the user-inputed LayeredMedia object. Relevant quantities are stored.
        
    Args:
         (): 
        
    Attributes:
        The following quantities are stored:
        - polarization of light
        - array of layers contained in LayeredMedia
        - entrance material (the environment before the first boundary of LayeredMedia
        - exit material (the environment after the last boundary of LayeredMedia
        - transfer matrix
        - a dictionary of matrices (propagation and transmission matrices)
        - a dictionary of layers (Layer or Surface objects)
    """
    
    def __init__(self,layeredMedia,polarization = 'p'):
        """Construct a TransferMatrix object.
        
        Args:
            layeredMedia (LayeredMedia): a material sample. 
            polarization (str): can be either 'p' or 's' to represent p-polarized and s-polarized light respectively.
        
        Return:
            void
        
        """
        self._check_polarization_(polarization)
        self.polarization = polarization
        
        self._layers = layeredMedia.get_layers()
        self.entrance = layeredMedia.get_entrance()
        self.exit = layeredMedia.get_exit()
        
        self.matrix = sympy.Matrix([[1,0],[0,1]])
        self.matrixDictionary = {}
        self.layerDictionary = {}
        
        self._index = 0
        self.layerIndex = 2     #The final value of layerIndex after building the material is two more than the actual number of layers. 
        self._surfaceCount = 1
        
        self.create_matrix()
    
    def _check_polarization_(self,polarization):
        """Ensure clean input of the polarization parameter. 
        
        This is a helper method.
        
        Args:
            polarization (str): the user-inputed polarization 
        
        Return:
            void
            
        Raises:
            ValueError: if polarization is not 's' or 'p'
        
        """
        if (polarization!='s' and polarization!='p'):
            Logger.raiseException('Invalid polarization. Accept only \'s\' or \'p\'.',exception=ValueError)
    
    def create_matrix(self):
        """Create a transfer matrix by unpacking the LayerMedia object into its constituent Layer and Surface objects and construct corresponding matrices.
        
        Unpack the LayeredMedia object into its constituents: Layer and Surface. Create a TransmissionMatrix object for Surface; create a PropagationMatrix for Layer.
        Add the layers and surfaces into layerDictionary (a class variable); add TransmissionMatirx and PropagationMatrix objects into matrixDictionary (a class variable).
        The resultant transfer matrix is stored in matrix (a class variable).
        Edge case 1: If no surface is present between two Layers, a Surface object with zero conductivity and its corresponding TransmissionMatrix is created.
        Edge case 2: If no surface is present at the boundaries, a Surface object with zero conductivity and its corresponding TransmissionMatrix is created.
        Edge case 3: a LayeredMedia object containing other LayeredMedia objects is addressed by recursion. 
        
        Args:
            None 
        
        Return:
            void
        
        """
        for layer in self._layers:
            
            if isinstance(layer,Surface):
                self._check_previous_layer_()
                self.layerDictionary["S{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = layer
                myTransmissionMatrix = TransmissionMatrix(self.polarization,self._surfaceCount)
                self.matrixDictionary["T{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = myTransmissionMatrix
                self.matrix *= myTransmissionMatrix.get_matrix()
                self._surfaceCount += 1
            
            #recursively unpack a LayeredMedia constituents into Layer and Surface
            elif isinstance(layer,LayeredMedia):
                myTransferMatrix = TransferMatrix(layer)
                self.matrix *= myTransferMatrix.get_matrix()
                
            elif isinstance(layer,Layer):
                preLayer = self._deepest_pre_layer_(self._layers[self._index-1])    
             
                if self._index == 0:     #override the marginal case in which the end of the material, which is accessed by self._layers(-1) is a surface.
                    preLayer = Layer(self.entrance,1)     #The thickness of the entrance material doesn't matter; it's arbitrarily set to 1 
                if isinstance(preLayer,Layer):      #The condition for two adjacent _layers; need to insert a transmission matrix in between
                    self._insert_transmission_matrix_()
                
                self.layerDictionary['L{}'.format(self.layerIndex)] = layer
                myPropagationMatrix = PropagationMatrix(self.layerIndex)
                self.matrixDictionary["P{}".format(self.layerIndex)] = myPropagationMatrix
                self.matrix *= myPropagationMatrix.get_matrix()
                self.layerIndex += 1
            
            #Create another Surface at the exit boundary if the end of the Layered Medium is a layer
            if (self._index+1==len(self._layers) and isinstance(layer,Layer)):
                self.layerDictionary["S{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = Surface()
                myTransmissionMatrix = TransmissionMatrix(self.polarization,self._surfaceCount)
                self.matrixDictionary["T{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = myTransmissionMatrix
                self.matrix *= myTransmissionMatrix.get_matrix()
                self._surfaceCount += 1
                
            self._index += 1
    
    def _check_previous_layer_(self):
        """Ensure the inputted LayerMedia does not have the occurrence of two adjacent Surface objects. 
        
        This is a helper method.
        
        Args:
            None
        
        Return:
            void
            
        Raises:
            ValueError: if LayeredMedia contains two adjacent Surface objects.
        
        """
        preLayer = self._deepest_pre_layer_(self._layers[self._index-1])
        if (self._index != 0 and isinstance(preLayer,Surface)):
            Logger.raiseException('Cannot have two adjacent Surface objects.',exception=ValueError)

    def _insert_transmission_matrix_(self):
        """Create a Surface with zero conductivity and insert a TransmissionMatrix object.  
        
        Physically, the material is not modified. The added Surface object (and its corresponding TransmissionMatrix) is a place holder.
        
        This is a helper method.
        
        Args:
            None
        
        Return:
            void
        
        """
        self.layerDictionary["S{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = Surface() 
        
        myTransmissionMatrix = TransmissionMatrix(self.polarization,self._surfaceCount)
        self.matrixDictionary["T{0}{1}".format(self._surfaceCount,self._surfaceCount+1)] = myTransmissionMatrix
        self.matrix *= myTransmissionMatrix.get_matrix()
        self._surfaceCount += 1
    
    def _deepest_pre_layer_(self,layer):
        """Recursively find the Layer object immediately before the specified Layer object.  
        
        This is a helper method. It addresses the case if the previous Layer object is contained in a hierarchy of LayeredMedia objects.
        
        Args:
            layer (Layer): the layer of interest
        
        Return:
            a Layer object immediately before the layer of interest
        
        """
        if isinstance(layer,LayeredMedia):
            layer = self._deepest_pre_layer_(layer.get_layers()[-1])        #need to choose the last Layer (i.e. -1 _index) embedded in the previous LayeredMedia
        return layer
    
    def _deepest_post_layer_(self,layer):
        """Recursively find the Layer object immediately after the specified Layer object.  
        
        This is a helper method. It addresses the case if the post Layer object is contained in a hierarchy of LayeredMedia objects.
        
        Args:
            layer (Layer): the layer of interest
        
        Return:
            a Layer object immediately after the layer of interest
        
        """
        if isinstance(layer,LayeredMedia):
            layer = self._deepest_pre_layer_(layer.get_layers()[0])     #need to choose the first Layer (i.e. 0 _index) embedded in the following LayeredMedia
        return layer

    def get_matrix(self): 
        """Get the transfer matrix of the user-inputed sample.  
        
        Args:
            None
        
        Return:
            matrix (a class variable)
        
        """
        return copy.copy(self.matrix)
    
    def get_matrix_dictionary(self): 
        """Get the dictionary of matrices (transmission and propagation matrices).  
        
        The key of the matrixDictionary is organized using the following scheme:
        For TransmissionMatrix,
            'T'+str(_index)+str(_index+1)
            e.g. 'T12' is the key for the first surface (between the LayeredMedia and the entrance boundary)
        For PropagationMatrix,
            'P'+str(_index)
            e.g. 'P2' is the key for the first layer
        
        The entrance material (usually Air) is assigned a _index of 1. Every _layers after increments from it.
        
        Args:
            None
        
        Return:
            matrixDictionary (a class variable)
        
        """
        return copy.copy(self.matrixDictionary)
    
    def get_layer_dictionary(self): 
        """Get the dictionary of _layers (Layer and Surface objects).  
        
        The key of the layerDictionary is organized using the following scheme:
        For Surface,
            str(_index)+str(_index+1)
            e.g. '12' is the key for the first surface (between the LayeredMedia and the entrance boundary)
        For Layer,
            str(_index)
            e.g. '2' is the key for the first layer
        
        The entrance material (usually Air) is assigned a _index of 1. Every _layers after increments from it.
        
        Args:
            None
        
        Return:
            layerDictionary (a class variable)
        
        """
        return copy.copy(self.layerDictionary)
    
    def get_layer_count(self):
        """Get the number of layers contained within the user-inputed LayeredMedia.
        
        Args:
            None
        
        Return:
            layerIndex (a class variable)
            
        """
        return copy.copy(self.layerIndex)
    
class TransmissionMatrix:
    """TransmissionMatrix class create analytical expression corresponding to a Surface object.
    
    Attributes:
        The following quantities are stored:
        - analytical expression of the transmission matrix
        
    """

    def __init__(self,polarization,surfaceCount, surfaceCurrent='default'):
        """Construct and store a TransmissionMatrix object.
        
        Args: 
            polarization (str): can be either 'p' or 's' to represent p-polarized and s-polarized light respectively.
            surfaceCount (int): the index of the Surface (Layer is not included in this counting)
            surfaceCurrent (str): can be either 'default' or 'self'. 
                If 'default', the full TransmissionMatrix (including contribution from the surface current) is calculated.
                if 'self', the surface current contribution is omitted.
        
        Return:
            void
        
        Raise:
            ValueError: if surfaceCurrent is not either 'default' or 'self'
            ValueError: if polarization is not either 'p' or 's'
        
        """
        if polarization == 'p':
            epsilon_in,epsilon_out,omega,kz_in,kz_out,sigma = sympy.symbols('epsilon_{0},epsilon_{1},omega,k_z{0},k_z{1},sigma{0}{1}'.format(surfaceCount,surfaceCount+1))
            
            eta_p = epsilon_in*kz_out/(epsilon_out*kz_in) 
            xi_p = 4*sympy.pi*(sigma/29979245368)*kz_out/(epsilon_out*omega)     #29979245368, i.e. c is the conversion factor to convert sheet conductivity (sigma) from unit of cm/s to dimensionless (as frequency is measured in wavenumbers).
            half = sympy.Rational(1,2)
                
            if surfaceCurrent == 'default':
                self.T = half*sympy.Matrix([[1+eta_p+xi_p, 1-eta_p-xi_p], [1-eta_p+xi_p, 1+eta_p-xi_p]])
                
            elif surfaceCurrent == 'self':
                self.T = half*sympy.Matrix([[1+eta_p, 1-eta_p], [1-eta_p, 1+eta_p]])
                
            else:
                Logger.raiseException('Wrong argument for the surfaceCurrent parameter. Accept only \'default\' or \'self\'.',exception=ValueError)
        
        elif polarization == 's':
            c,kz_in,kz_out,sigma,mu_in,omega = sympy.symbols('c,k_z{0},k_z{1},sigma{0}{1},mu_{0},omega'.format(surfaceCount,surfaceCount+1))
            
            eta_s = kz_out/kz_in
            xi_s = 4*sympy.pi*(sigma/29979245368)*mu_in*omega/kz_in     #29979245368, i.e. c is the conversion factor to convert sheet conductivity (sigma) from unit of cm/s to dimensionless (as frequency is measured in wavenumbers).
            half = sympy.Rational(1,2)
            self.T = half*sympy.Matrix([[1+eta_s+xi_s, 1-eta_s+xi_s], [1-eta_s-xi_s, 1+eta_s-xi_s]])
            
        else:
            Logger.raiseException('Cannot create transmission matrix due to invalid polarization. Accept only \'s\' or \'p\'.',exception=ValueError)
    
    def get_matrix(self): 
        """Get the transmission matrix of the corresponding Surface object.  
        
        Args:
            None
        
        Return:
            T (a class variable)
        
        """
        return copy.copy(self.T)

class PropagationMatrix:
    """PropagationMatrix class create analytical expression corresponding to a Layer object.
    
    Attributes:
        The analytical expression of the following quantity is stored:
        - propagation matrix
    
    """

    def __init__(self,layerCount):
        """Construct and store a PropagationMatrix object.
        
        Args:
            layerIndex (int): the index of the Layer (Surface is not included in this counting)
        
        Return:
            void
        
        """
        z,kz = sympy.symbols('z{0},k_z{0}'.format(layerCount))
        self.P = sympy.Matrix([[sympy.exp(-sympy.I*kz*z), 0], [0, sympy.exp(sympy.I*kz*z)]])
        
    def get_matrix(self): 
        """Get the propagation matrix of the corresponding Layer object.  
        
        Args:
            None
        
        Return:
            P (a class variable)
        
        """
        return copy.copy(self.P)
    
class CurrentDensityVector:
    """CurrentDensityVector class create analytical expression of surface current based on the continuity equation.
        
    Attributes:
        The analytical expression of the following quantity is stored:
        - surface current in terms of charge density (rho)
    """

    def __init__(self):
        """Construct and store a CurrentDensityVector object.
        
        Args:
            None
        
        Return:
            void
        
        """
        rho,q,omega = sympy.symbols('rho,q,omega')
        self.J = sympy.Matrix([[-rho*omega/q],[-rho*omega/q]])
    
    def get_vector(self): 
        """Get the analytical expression for surface current in terms of charge density (rho).  
        
        Args:
            None
        
        Return:
            J (a class variable)
        
        """
        return copy.copy(self.J)
    
