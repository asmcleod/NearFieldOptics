from NearFieldOptics import Materials as mat
import numpy
from matplotlib.pyplot import *
from common import plotting

chemical_potentials=range(600,1100,100)
qs,omegas=numpy.ogrid[0:.1:.001,500:1500:5] #in nm^-1, cm^-1

qs1d=qs[:,0]
omegas1d=omegas[0]

for i,mu in enumerate(chemical_potentials):
    mat.SingleLayerGraphene.chemical_potential=mu
    rp=mat.SingleLayerGraphene.reflection_p(omegas,qs*1e7)
    figure()
    
    print qs1d.shape,omegas1d.shape,rp.imag.shape
    contourf(qs1d,omegas1d,numpy.transpose(rp.imag),50)
    title('$\mu=%i\,[cm^{-1}]$'%mu,fontsize=19)