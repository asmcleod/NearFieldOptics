

import math
import numpy
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import scipy
import scipy.special
import scipy.optimize
warnings.simplefilter('default', DeprecationWarning)


def sqr(z):
  return z*z

def faddeeva(z, NT=None):
  """computes w(z) = exp(-z^2) erfc(iz) according to
  J. A. C. Weideman, "Computation of the Complex Error Function,"
  NT: number of terms to evaluate in approximation
      Weideman claims that NT=16 yields about 6 digits,
      NT=32 yields about 12 digits, for Im(z) >= 0.
      However, faddeeva(100.,16) yields only 8 accurate digits.
  For Im(z)>=0:
  By graphing, we see that NT=16 does yield errors < 10^{-6}
  and NT=32 errors < 10^{-12}
  NT=36 yields errors < 10^{-14}.
  For NT>=40, the errors will never be (everywhere) smaller
  than 10^{-14}...values oscillate as NT increases above 40.

  For Im(z)<0, the relative accuracy suffers near zeros
  of w(z).
  """
  if (type(z) != numpy.ndarray):
    return faddeeva(numpy.array([z]), NT)
  if (NT==None):
    NT = 42
  numSamplePts = 2*NT
  ind = numpy.arange(-numSamplePts+1.,numSamplePts)
  L = numpy.sqrt(NT/numpy.sqrt(2))
  theta = (math.pi / numSamplePts) * ind
  t = L * numpy.tan(0.5*theta)
  fn = numpy.empty((ind.size+1,),dtype = t.dtype)
  fn[0] = 0.
  fn[1:] = numpy.exp(-t*t) * (L*L + t*t)
  polyCoefs = numpy.fft.fft(numpy.fft.fftshift(fn)).real / (2*numSamplePts)
  polyCoefs = numpy.flipud(polyCoefs[1:(NT+1)])
  
  negzInd = z.imag < 0
  signs = numpy.ones(z.shape, dtype=z.dtype)
  signs[negzInd] *= -1.
    
  # first use of z follows
  polyArgs = (L + (0.+1.j)*signs*z)/(L-(0.+1.j)*signs*z)
  polyVals = numpy.polyval(polyCoefs,polyArgs)
  res = 2.*polyVals/ sqr(L-(0.+1.j)*signs*z) \
    + (1./math.sqrt(math.pi)) / (L-(0.+1.j)*signs*z)
  res *= signs
  res[negzInd] += 2.*numpy.exp(-z[negzInd]*z[negzInd])
  
  #res=res.squeeze()
  if not res.ndim: res=res.tolist()
  
  return res
 
def graphFaddeevaAccuracy(nt1, nt2):
  import pylab
  xmin = 0.01
  xmax = 100.3
  nx = 437
  ymin = 0.0099
  ymax = 100.1
  ny = 439
  logxmin = numpy.log10(xmin)
  logxmax = numpy.log10(xmax)
  logymin = numpy.log10(ymin)
  logymax = numpy.log10(ymax)
  xs = numpy.arange(logxmin,logxmax,(logxmax-logxmin)/nx)
  ys = numpy.arange(logymin,logymax,(logymax-logymin)/ny)
  [X,Y] = numpy.meshgrid(xs,ys)
  # now X[i,j] = xs[j], Y[i,j] = ys[i]
  Z = numpy.power(10.,X) + 1.j*numpy.power(10.,Y)
  val1 = faddeeva(Z,nt1)
  val2 = faddeeva(Z,nt2)
  relErr = numpy.log10(abs(1. - abs(val2/val1)))
  relErr[relErr<-16.] = -16.
  minErr = numpy.min(relErr)
  maxErr = numpy.max(relErr)
  print("rel diff (between NT=", nt1, "and NT=", nt2, ") ranges between 10^" + str(maxErr) + " and 10^" + str(minErr))
  numC = int(math.ceil(maxErr)-math.floor(minErr))
  cp = pylab.contourf(X,Y,relErr,numC)
  #pylab.clabel(cp, inline=1)
  cb = pylab.colorbar(cp)
  pylab.xlabel('log10(Re[z])')
  pylab.ylabel('log10(Im[z])')
  pylab.title('log10(relative difference)')
  
  pylab.show()
  
  
if __name__ == "__main__":
  graphFaddeevaAccuracy(42,100)
