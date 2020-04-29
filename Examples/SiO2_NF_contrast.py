# -*- coding: utf-8 -*-
# This example shows calculation of the reflection coefficient and
#   near-field signal from a thin film of silicon dioxide on silicon
#   as would be measured by a near-field probe of different tip radii.
#   It will be seen that contrast of SiO2 is boosted for smaller tip radii,
#   since the associated near-fields will be less capable of penetrating
#   "through" the SiO2 layer.

from  numpy import *
from matplotlib.pyplot import *

from NearFieldOptics import Materials as Mat
from NearFieldOptics import TipModels as T

########################################################
#--- 1. Define the materials and make a layered structure:
#       This will be 300nm of silicon dioxide on a doped
#       silicon substrate.
########################################################

subs = Mat.Si_Doped #This is a flexible model for silicon with variable doping level
SiO2 = Mat.SiO2_300nm #One of several models for silicon dioxide in the mid-infrared

SiO2_thickness = 300e-7; # SiO2 thickness 300 nm

#Construct a layered structure which is simply SiO2 of designated
#   thickness on top of the silicon substrate.
layers_SiO2 = Mat.LayeredMedia((SiO2,SiO2_thickness),exit = subs)

##########################################################
#--- 2. Compute optical response of layered structure:
#       We'll look at both the reflection coefficient
#       in the near-field regime (quasi-electrostatic
#       optical response, in the limit of high in-plane
#       momentum of light) as well as the optical contrast
#       that would emerge in a SNOM experiment.
##########################################################

#mid-IR frequencies (in units of cm^-1)
frequencies = linspace(1065,1205,71) 
a=30 #radius of tip apex for a near-field probe, in nanometers
q_a=1/a #This defines the characteristic in-plane momentum for near-fields in SNOM

#Compute (p-polarized) reflection coefficient at designated momentum
rp_nf_SiO2 = layers_SiO2.reflection_p(frequencies,q=q_a)

S_SiO2_dict={}
for a in [10,20,30,50]:
    print 'Computing SNOM contrast for tip radius a=%i...'%a
    S_SiO2_dict[a] = T.LightningRodModel(frequencies,rp=layers_SiO2.reflection_p,a=a,amplitude=40,\
                                          normalize_to=Mat.Au.reflection_p,normalize_at=1000)

##########################################################
#--- 3. Plot the results:
#       The computed arrays are of type `ArrayWithAxes`
#       and have a method `.plot(...)` which automatically
#       plots the array against its intrinsic axes.
##########################################################

figure()
abs(rp_nf_SiO2).plot()
ylabel('Reflection coefficient (p-polarized)')

figure();
for a in S_SiO2_dict.keys():
    S_SiO2=S_SiO2_dict[a]
    abs(S_SiO2['signal_3']).plot(label = '%i nm'%a)
    
ylabel('s_3(SiO2)/s_3(Au)')
title('285 nm SiO2 on Si')
legend(loc='best')

show()