import numpy
import scipy
from functools import wraps

##Make sure all our vectorization inherits keyword signatures, etc##
def vectorize_matrix_func(matrix_func):
    """Return a function that will take inputs, compare them,
    and broadcast matrix inputs over array inputs before
    performing the indicated operation."""
        
    @wraps(matrix_func)
    def vec_matrix_func(*args):
        
        #If no array inputs, pass (single number?) inputs through
        args_are_arrays=[isinstance(arg,numpy.ndarray) \
                         and not isinstance(arg,numpy.matrix) \
                         for arg in args]
        if True not in args_are_arrays:
            new_args=[arg.astype(numpy.complex) if isinstance(arg,numpy.matrix) \
                      else arg for arg in args]
            return matrix_func(*new_args)
        
        #Otherwise apply array shape to existent matrices
        #(to make sure operations with arrays don't get botched in the vectorization)
        args_are_matrices=[isinstance(arg,numpy.matrix) \
                           for arg in args]
        shape=args[args_are_arrays.index(True)].shape
        
        new_args=[]
        for i,arg in enumerate(args):
            if args_are_matrices[i]:
                new_arg=numpy.ndarray(shape,dtype=numpy.object)
                new_arg.fill(arg)
            else: new_arg=arg
            new_args.append(new_arg)
            
        #Run the vectorized function over arrays of matrices, etc.
        #Tested to work if dtype of each arg array is matrix
        #otype=object, since we don't know what `matrix func` will spit out (probably another matrix?)
        result=numpy.vectorize(matrix_func,otypes=[numpy.object])(*new_args)
        if not isinstance(result.flat[0],numpy.ndarray):
            result=result.astype(complex)
        
        return result
    
    return vec_matrix_func
                             

def elements_into_2x2matrices(xx=0,xy=0,\
                              yx=0,yy=0):
    
    return numpy.matrix([[xx,xy],\
                         [yx,yy]]).astype('complex')
                         
elements_into_2x2matrices=numpy.vectorize(elements_into_2x2matrices,otypes=[object])

def elements_into_3x3matrices(xx=0,xy=0,xz=0,\
                              yx=0,yy=0,yz=0,\
                              zx=0,zy=0,zz=0):
    
    return numpy.matrix([[xx,xy,xz],\
                         [yx,yy,yz],\
                         [zx,zy,zz]]).astype('complex')
                         
elements_into_3x3matrices=numpy.vectorize(elements_into_3x3matrices,otypes=[object])

def elements_into_2vectors(x=0,y=0):
    
    return numpy.matrix([[x],[y]])
                         
elements_into_2vectors=numpy.vectorize(elements_into_2vectors,otypes=[object])

def elements_into_3vectors(x=0,y=0,z=0):
    
    return numpy.matrix([[x],[y],[z]])
                         
elements_into_3vectors=numpy.vectorize(elements_into_3vectors,otypes=[object])
    
def elements_into_diag_3x3matrices(xx=0,yy=0,zz=0):
    
    return elements_into_3x3matrices(xx,0,0,\
                                     0,yy,0,\
                                     0,0,zz)


_3x3matrix_element_listings_=dict(xx=(0,0),xy=(0,1),xz=(0,2),\
                                   yx=(1,0),yy=(1,1),yz=(1,2),\
                                   zx=(2,0),zy=(2,1),zz=(2,2))
@vectorize_matrix_func
def elements_from_3x3matrices(matrix,element='xx'):
    
    return matrix[_3x3matrix_element_listings_[element]]

_2vector_element_listings_=dict(x=0,y=1)
@vectorize_matrix_func
def elements_from_2vectors(vector,element='x'):
    
    return vector[_2vector_element_listings_[element],0]

_3vector_element_listings_=dict(x=0,y=1,z=2)
@vectorize_matrix_func
def elements_from_3vectors(vector,element='x'):
    
    return vector[_3vector_element_listings_[element],0]


@vectorize_matrix_func
def inv_3x3(M):
    
    #Suppose it's not a matrix but simply a constant (multiple of identity)
    if not isinstance(M,numpy.matrix): return 1/M
    
    #from http://ardoris.wordpress.com/2008/07/18/general-formula-for-the-inverse-of-a-3x3-matrix/
    a=M[0,0]; b=M[0,1]; c=M[0,2]
    d=M[1,0]; e=M[1,1]; f=M[1,2]
    g=M[2,0]; h=M[2,1]; i=M[2,2]
    
    pref=numpy.complex(a*(e*i-f*h)+\
                       b*(f*g-d*i)+\
                       c*(d*h-e*g))
    Mp=numpy.matrix([[e*i-f*h,c*h-b*i,b*f-c*e],\
                     [f*g-d*i,a*i-c*g,c*d-a*f],\
                     [d*h-e*g,b*g-a*h,a*e-b*d]])
    
    #print 'O:',M
    #print 'pref:',pref
    return 1/pref*Mp

reduction_factor=1e12

def get_O_matrix(kz,q,omega,eps,mu):
    
    kvec=numpy.matrix([q,0,kz[0]+kz[1]*1j]).T
    O=omega**2*eps*mu+kvec*kvec.T-numpy.sum(numpy.array(kvec)**2)*numpy.matrix(numpy.eye(3))
    O=O.astype('complex')
    
    return O/reduction_factor

def detO(kz,q,omega,eps,mu):
    
    #print 'kz:',kz
    O=get_O_matrix(kz,q,omega,eps,mu)
    D=numpy.linalg.det(O)
    
    return D.real,D.imag

def detO_gradient(kz,q,omega,eps,mu):
    
    O=get_O_matrix(kz,q,omega,eps,mu)
    detO=numpy.linalg.det(O)
    Oinv=numpy.array(inv_3x3(O))
    dOdkz1=numpy.array([[2*(kz[0]+kz[1]*1j),    0,                      q],\
                        [0,                     -2*(kz[0]-kz[1]*1j),    0],\
                        [q,                     0,                      0]])
    dOdkz2=numpy.array([[2*(kz[1]-kz[0]*1j),    0,                      q*1j],\
                        [0,                     2*(kz[1]-kz[0]*1j),     0],\
                        [q*1j,                  0,                      0]])
    sum1=numpy.sum(numpy.diag(Oinv*dOdkz1))
    sum2=numpy.sum(numpy.diag(Oinv*dOdkz2))
    
    grad=numpy.array([2*numpy.real(detO*sum1*numpy.conj(detO)),\
                      2*numpy.real(detO*sum2*numpy.conj(detO))])/(2*numpy.abs(detO))
    
    return grad