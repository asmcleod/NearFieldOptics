# -*- coding: utf-8 -*-
# This example shows a simple calculation of SNOM contrast between two bulk materials: silicon and gold.

from  numpy import *
from matplotlib.pyplot import *

from NearFieldOptics import Materials as Mat
from NearFieldOptics import TipModels as T


##########################################################
#--- 1. Compute SNOM contrast of silicon relative to gold:
#       This uses `T.LightningRodModel` (the celebrated model
#       for SNOM, c.f. Phys. Rev. B 90, 085136.) to solve the
#       quasi-electrostatic scattering problem of a tip over
#       a planar sample and compute the scattered field and
#       demodulated "near-field signal".
##########################################################

#mid-IR frequencies (in units of cm^-1)
frequencies = linspace(700,1000,100)

#The call signature for `T.LightningRodModel` is explicated below,
#   showing the meaning (in order) of the keyword arguments:
# T.LightningRodModel(frequency, rp, tip radius, number q pts,
#                       number z pts, tapping amplitude, normalization material, normalization frequency)
S_lay_Si = T.LightningRodModel(frequencies,rp=Mat.Si.reflection_p,a=30,Nqs=244,\
              Nzs=40,amplitude=80,normalize_to=Mat.Au.reflection_p,normalize_at=1000)

##########################################################
#--- 2. Plot the results:
#       The computed arrays are of type `ArrayWithAxes`
#       and have a method `.plot(...)` which automatically
#       plots the array against its intrinsic axes.
##########################################################

figure();
abs(S_lay_Si['signal_3']).plot()
#The result is implicitly normalized to the signal from gold,
#    because of `normalize_to=...` in the calculation above.
ylabel('s_3(Si)/s_3(Au)')
show()