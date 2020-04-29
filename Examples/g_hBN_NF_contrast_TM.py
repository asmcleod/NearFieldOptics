# -*- coding: utf-8 -*-
#   This example shows construction of a graphene heterostructure comprising a 
#   monolayer of graphene sandwiched between two thin layers of hexagonal boron
#   nitride (hBN), all sitting on a silicon substrate with topped by a silicon
#   oxide layer.  We compute the reflection coefficient of this structure in the
#   energy range of graphene plasmons, and compute the near-field response that
#   would be detected in a SNOM experiment.

from  numpy import *
from matplotlib.pyplot import *

from NearFieldOptics import Materials as Mat
from NearFieldOptics import TipModels as T

########################################################
#--- 1. Build multilayer structure:
#       This will be boron nitride sandwiching graphene,
#       all on top of silicon dioxide, with silicon 
#       substrate (infinite half-space) underneath.
########################################################

#Choose which boron nitride material definition to use
BN = Mat.BN_GPR

#Make an instance of graphene monolayer (constructor is called `SingleLayerGraphene`)
graphene = Mat.SingleLayerGraphene(chemical_potential=2400,gamma=30)

#Choose which silicon dioxide material definition to use
SiO2 = Mat.SiO2_300nm
subs = Mat.Si

#Define thickness of boron nitride layers
BN1_thickness = 8.5e-7; # top BN thickness 8.5 nm in units of cm
BN2_thickness = 23e-7; # bottom BN thickness 23 nm

#Construct two layered structures, one with graphene in the sandwich and one without
layers_BN = Mat.LayeredMediaTM((BN,BN1_thickness),(BN,BN2_thickness),\
                             (SiO2,285e-7),exit = subs)
layers_gBN = Mat.LayeredMediaTM((BN,BN1_thickness),graphene,(BN,BN2_thickness),\
                              (SiO2,285e-7),exit = subs)

##########################################################
#--- 2. Compute optical response of structures:
#       We'll look at both the reflection coefficient
#       in the near-field regime (quasi-electrostatic
#       optical response, in the limit of high in-plane
#       momentum of light) as well as the optical contrast
#       that would emerge in a SNOM experiment.
##########################################################

#Define parameters for optical response of the sample
frequencies = linspace(700,1000,100) #mid-IR frequencies (in units of cm^-1)
a=20# #radius of tip apex for a near-field probe, in nanometers
q_a=1/a #This defines the characteristic in-plane momentum for near-fields in SNOM

#Compute (p-polarized) reflection coefficient at designated momentum
rp_nf_BN = layers_BN.reflection_p(frequencies,q=q_a)
rp_nf_gBN = layers_gBN.reflection_p(frequencies,q=q_a)

#Compute near-field contrast relative to gold (`Mat.Au`) for the structures
#   The result will be a dictionary with several arrays evaluated over the `frequencies`
S_lay_BN = T.LightningRodModel(frequencies,rp=layers_BN.reflection_p,a=a,Nqs=244,\
              Nzs=40,amplitude=80,normalize_to=Mat.Au.reflection_p,normalize_at=1000)
S_lay_gBN = T.LightningRodModel(frequencies,rp=layers_gBN.reflection_p,a=a,Nqs=244,\
           Nzs=40,amplitude=80,normalize_to=Mat.Au.reflection_p,normalize_at=1000)

##########################################################
#--- 3. Plot the results:
#       The computed arrays are of type `ArrayWithAxes`
#       and have a method `.plot(...)` which automatically
#       plots the array against its intrinsic axes.
##########################################################

#Plot reflection coefficient
figure();
abs(rp_nf_BN).plot(label=r'$\beta$ for hBN')
abs(rp_nf_gBN).plot(label=r'$\beta$ for G+hBN')
ylabel(r'$\beta$ (reflection coeff.)')
legend()

#Plot near-field optical contrast versus gold
figure();
abs(S_lay_BN['signal_3']).plot(label=r'$S_3$ for hBN')
abs(S_lay_gBN['signal_3']).plot(label=r'$S_3$ for G+hBN')
ylabel('s_3/s_3(Au)')
legend()

#Plot the relative change in near-field contrast for these two
# structures by the introduction of graphene into the "sandwich"
figure();
abs(S_lay_gBN['signal_3']/S_lay_BN['signal_3']).plot()
ylabel(r'$S_3($G+hBN$)/S_3($hBN$)$')

show()