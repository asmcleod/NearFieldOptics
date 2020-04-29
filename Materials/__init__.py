from .material_types import *
from .TransferMatrixMedia import *

#########
#---Al2O3
#########

#E refers to dipole oscillations perpendicular to c-axis (ordinary)
E_TO_ws=[385,439.1,569,633.63]; E_LO_ws=[387.6,481.68,629.5,906.6]
E_TO_gs=[3.3,3.1,4.7,5]; E_LO_gs=[3.1,1.9,5.9,14.7]
damp_factor=5
E_TO_gs=[E_TO_g*damp_factor for E_TO_g in E_TO_gs]
E_LO_gs=[E_LO_g*damp_factor for E_LO_g in E_LO_gs]

#A refers to dipole oscillations parallel to c-axis (extraordinary)
A_TO_ws=[397.52,582.41]; A_LO_ws=[510.87,881.1]
A_TO_gs=[5.3,3]; A_LO_gs=[1.1,15.4]
A_TO_gs=[A_TO_g*damp_factor for A_TO_g in A_TO_gs]
A_LO_gs=[A_LO_g*damp_factor for A_LO_g in A_LO_gs]

eps_inf=[3.072]*2+[3.077]
phonon_params=[[(E_LO_w,E_LO_g,E_TO_w,E_TO_g) for \
                E_LO_w,E_LO_g,E_TO_w,E_TO_g in zip(E_LO_ws,E_LO_gs,E_TO_ws,E_TO_gs)]]*2+\
              [[(A_LO_w,A_LO_g,A_TO_w,A_TO_g) for \
                A_LO_w,A_LO_g,A_TO_w,A_TO_g in zip(A_LO_ws,A_LO_gs,A_TO_ws,A_TO_gs)]]
Al2O3=AnisotropicMaterial(eps_infinity=eps_inf,\
                          phonon_params=phonon_params)

ev_to_hz=1/(4.136e-15) #divide by planch's constant h
ev_to_wn=ev_to_hz/(3e10) #divide by speed of light
ws_o=[w_o*ev_to_wn for w_o in [.0477,.0548,.0705,.0781]]
ws_e=[w_e*ev_to_wn for w_e in [.0496,.0723,.0810]]
ss_o=[s_o*w_o**2 for s_o,w_o in zip([.3,2.6,2.55,.4], ws_o)]
ss_e=[s_e*w_e**2 for s_e,w_e in zip([6.22,2,.09], ws_e)]
gs_o=[g_o*w_o for g_o,w_o in zip([.08,.45,.026,.04], ws_o)]
gs_e=[g_e*w_e for g_e,w_e in zip([.054,.035,.06], ws_e)]

damping_factor=5
eps_lps=[[(s,w,g*damping_factor) for s,w,g in zip(ss_o,ws_o,gs_o)]]*2+\
        [[(s,w,g*damping_factor) for s,w,g in zip(ss_e,ws_e,gs_e)]]
eps_inf=[3.064]*2+[3.038]
Al2O3_2=AnisotropicMaterial2(eps_infinity=eps_inf,\
                            eps_lps=eps_lps)

######
#---Au
######

#Ordal - Applied Optics - 1985
#plasma_f=7.28e4
#damping_f=215

#Blaber
plasma_f=6.9e4
damping_f=148.4

Au=IsotropicMaterial(eps_infinity=1,drude_params=(plasma_f,damping_f))

##########################
#---Bi2Se3: Bulk tabulated
##########################
#
Bi2Se3_Bulk=TabulatedMaterialFromFile('Bi2Se3_epsilon.pickle')
#Bi2Se3_15QL=TabulatedMaterialFromFile('Bi2Se3_15QL_epsilon.pickle')
#Bi2Se3s={6:Bi2Se3_Bulk,\
#         15:Bi2Se3_Bulk,\
#         30:Bi2Se3_Bulk,\
#         60:Bi2Se3_Bulk}

######
#---BN
######

#Geick, Perry, & Rupprecht 1966 "Normal Modes of Hexagonal Boron Nitride"
Amps=[1.23e5,3.49e6]; w_TOs=[767,1367]; w_LOs=[778,1610]; gs=[35,29]
eps_lps=[[(Amp,w_TO,g) for Amp,w_TO,g in zip(Amps,w_TOs,gs)]]*2
#Amps=[3.26e5,1.04e6]; w_TOs=[783,1510]; w_LOs=[828,1595]; gs=[80,80]
#eps_lps.append([(Amp,w_TO,g) for Amp,w_TO,g in zip(Amps,w_TOs,gs)])
Amps=[3.26e5]; w_TOs=[783]; w_LOs=[828]; gs=[80]
eps_lps.append([(Amp,w_TO,g) for Amp,w_TO,g in zip(Amps,w_TOs,gs)])
BN_GPR=AnisotropicMaterial(eps_infinity=[4.95,4.95,4.1],\
                     eps_lps=eps_lps)


w_TOs=[1403]; w_LOs=[1583]; gs=[63]; eps0=3.5
eps_lps=[[(eps0*(w_LO**2-w_TO**2),w_TO,g) for w_TO,w_LO,g in zip(w_TOs,w_LOs,gs)]]*2
w_TOs=[770]; w_LOs=[815]; gs=[28]; eps0=3.5
eps_lps+=[[(eps0*(w_LO**2-w_TO**2),w_TO,g) for w_TO,w_LO,g in zip(w_TOs,w_LOs,gs)]]
BN_STW=AnisotropicMaterial(eps_infinity=[4.95,4.95,4.1],\
                           eps_lps=eps_lps)


w_TOs=[1360]; w_LOs=[1614]; gs=[7]; eps_inf=2.95; eps0=6.9
eps_lps=[[(eps_inf*(w_LO**2-w_TO**2),w_TO,g) for w_TO,w_LO,g in zip(w_TOs,w_LOs,gs)]]*2
w_TOs=[760]; w_LOs=[825]; gs=[2]; eps_inf=4.9; eps0=3.48
eps_lps+=[[(eps_inf*(w_LO**2-w_TO**2),w_TO,g) for w_TO,w_LO,g in zip(w_TOs,w_LOs,gs)]]
BN_Caldwell=AnisotropicMaterial(eps_infinity=[2.95,2.95,4.9],\
                           eps_lps=eps_lps)

##############
#---Forsterite
##############

#Sogawa et al. - 2006 - "Infrared reflection spectra of forsterite crystal"

damping_factor=1
eps_inf={1:2.71,2:2.66,3:2.77}
wps_over_ws_sq={1:[.62,.45,.16,1.02,.42,.026,1.26,.074],\
                2:[.0071,.4,.11,.18,.042,.19,.28,.47,1.21,1.65,.11,.075],\
                3:[.33,.17,.0029,.23,.35,1.56,.95,.079,.32,.054,.027]}
ws={1:[874,502.2,476.3,415.1,408.4,304.5,291.2,276.1],\
    2:[984.2,872.5,838.8,526.5,505.9,455.8,416.4,395,348.2,288.6,276.5,143.4],\
    3:[976.4,957.9,841,603.8,501.5,402.6,379.1,319.5,294,274.6,200.8]}
gs_over_ws={1:[.0058,.021,.017,.015,.013,.014,.011,.0093],\
            2:[.0053,.0055,.0096,.025,.012,.02,.015,.014,.017,.016,.012,.014],\
            3:[.0064,.004,.0085,.021,.025,.015,.015,.0084,.012,.011,.014]} #Original

#Build forsterites with all possible non-identical pairs of axes (6 pairs)
Forsterites={}
for i in [1,2,3]:
    for j in [1,2,3]:
        if i==j: continue
        
        eps_lps=[[(wp_over_w_sq*w**2,w,g_over_w*w*damping_factor) for wp_over_w_sq,w,g_over_w \
                  in zip(wps_over_ws_sq[i],ws[i],gs_over_ws[i])]]*2+\
                [[(wp_over_w_sq*w**2,w,g_over_w*w*damping_factor) for wp_over_w_sq,w,g_over_w \
                  in zip(wps_over_ws_sq[j],ws[j],gs_over_ws[j])]]
        Forsterites[(i,j)]=AnisotropicMaterial(eps_infinity=[eps_inf[i]]+[eps_inf[j]]*2,\
                                               eps_lps=eps_lps)

## Uniaxial version of Forsterite ##
# Homogenize axes 1 and 2 from Sogawa into ordinary optical axes (weight each by 1/2) #
eps_lps=[[(wp_over_w_sq*w**2/2.,w,g_over_w*w*damping_factor) for wp_over_w_sq,w,g_over_w \
                  in zip(wps_over_ws_sq[1]+wps_over_ws_sq[2],\
                         ws[1]+ws[2],\
                         gs_over_ws[1]+gs_over_ws[2])]]+\
        [[(wp_over_w_sq*w**2/2.,w,g_over_w*w*damping_factor) for wp_over_w_sq,w,g_over_w \
                  in zip(wps_over_ws_sq[3],ws[3],gs_over_ws[3])]]
Forsterite=UniaxialMaterial(eps_infinity=[(eps_inf[1]+eps_inf[2])/2.,eps_inf[3]],\
                            eps_lps=eps_lps,drude_params=[])

## `AnisotropicMaterial2` needs to be debugged and possibly removed ##
#eps_lps=[[(wp_over_w_sq*w**2,w,g_over_w*w*damping_factor) for wp_over_w_sq,w,g_over_w \
#                  in zip(wps_over_ws_sq[i],ws[i],gs_over_ws[i])] for i in [1,2,3]]
#Forsterite=AnisotropicMaterial2(eps_infinity=[eps_inf[i] for i in [1,2,3]],\
#                             eps_lps=eps_lps,\
#                             drude_params=[])

#########
#---FePO4
#########

#Include TO modes 44 onwards
#XX Modes 45,55,61,66,70
TOs_a=[533.6287,    694.2418,   972.2623,   1070.617-35,   1117.326]
gs_a=[10,           10,         10,         10,         10]
Amps_a=[.35726,     .28559,     .10169,     1.04915,    .1388]

#YY Modes 44,57
TOs_b=[520.3665,    931.2895-35]
gs_b=[10,           10]
Amps_b=[.73558,     .77944]

#ZZ Modes 46,49,53,62,67,72
TOs_c=[537.9886,    591.5726,   658.2125,   978.0455,   1073.087-35,   1258.825]
gs_c=[10,           10,         10,         10,         10,         10]
Amps_c=[.48143,     .08065,     .34703,     .06206,     .675,       .00635]

eps_inf=[3.8503,3.3596,3.5142]
damping_factor=4
eps_lps=[[(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_a,Amps_a,numpy.array(gs_a)*damping_factor)],\
         [(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_c,Amps_c,numpy.array(gs_c)*damping_factor)],\
         [(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_b,Amps_b,numpy.array(gs_b)*damping_factor)]]
FePO4=AnisotropicMaterial(eps_infinity=eps_inf,\
                          eps_lps=eps_lps)
FePO4_2=AnisotropicMaterial2(eps_infinity=eps_inf,\
                             eps_lps=eps_lps)

###########
#---LiFePO4
###########

damping_factor=1.5

eps0_a=2.56
ws_a=       numpy.array([1093,      1027,   946.2,  644.3,  574,    476.8])
gs_a=       numpy.array([6.6,       10.9,   7.1,    3.15,   9.88,   43.35])
strengths_a=numpy.array([.015,      .512,   .009,   .051,   .25,    .326])#*eps0_a
amps_a=strengths_a*ws_a**2#*gs_a

eps0_b=2.84
ws_b=       numpy.array([929.3,     673.9,  545.9,  461.1,  417.9])
gs_b=       numpy.array([.5*15.3,       5.7,    8.75,   20,     38])
strengths_b=numpy.array([.4*.75,       .035,   .29,    .47,    .75])#*eps0_b
amps_b=strengths_b*ws_b**2#*gs_b

eps0_c=2.54
ws_c=       numpy.array([1070,      630.4,  494,    400])
gs_c=       numpy.array([16,        7,      38.4,   21])
strengths_c=numpy.array([.4,     .13,    .61,    .5])#*eps0_c
amps_c=strengths_c*ws_c**2#*gs_c

LiFePO4=AnisotropicMaterial2(eps_infinity=[eps0_a,eps0_c,eps0_b],\
                             eps_lps=[list(zip(amps_a,ws_a,gs_a*damping_factor)),
                                      list(zip(amps_c,ws_c,gs_c*damping_factor)),\
                                      list(zip(amps_b,ws_b,gs_b*damping_factor))])

#Include TO modes 44 onwards
#XX Modes 52,56,61,67,70,79,83
#TOs_a=[483.7513,    530.09,     595.7659,   678.3087,   961.1095,   1052.7989,  1127.1822]
TOs_a=[465.755,     522.154,    590.047,    671.772,    950.963,    1042.07,    1116.02]
gs_a=[10,           10,         10,         10,         10,         10,          10]
Amps_a=[.2267,      .0795,      .1299,      .0802,      .0029,      .4715,      .0148]

#YY Modes 44,46,54,57,71
#TOs_b=[388.9262,    441.9146,   498.6956,   554.8774,   961.9005]
TOs_b=[378.076,     412.614,    488.414,    551.104,    953.966]
gs_b=[10,           10,         10,         10,         10]
Amps_b=[.2305,      .7472,      .391,       .1188,      .6829]

#ZZ Modes 53,55,60,65,69,81,84
#TOs_c=[495.1698,    516.2701,   588.5594,   657.3456,   960.1293,   1103.5182,  1168.4879]
TOs_c=[470.51,      507.03,     587.624,    650.03,     949.5615,   1092.56,    1158.02]
gs_c=[10,           10,         10,         10,         10,         10,          10]
Amps_c=[.52709,     .155,      .02078,      .16174,     .00099,     .34732,     .03473]

eps_inf=[2.5512,2.6193,2.5877]
damping_factor=1
eps_lps=[[(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_a,Amps_a,numpy.array(gs_a)*damping_factor)],\
         [(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_c,Amps_c,numpy.array(gs_c)*damping_factor)],\
         [(Amp*w_TO**2,w_TO,g) for w_TO,Amp,g in zip(TOs_b,Amps_b,numpy.array(gs_b)*damping_factor)]]
LiFePO4_calc=AnisotropicMaterial2(eps_infinity=eps_inf,\
                                  eps_lps=eps_lps)

PMMA=TabulatedMaterialFromFile('PMMA_epsilon.pickle')

#################
#---PZT (Michael)
#################

class _PZT_(IsotropicMaterial):
    
    def __init__(self,*args,**kwargs):
        
        IsotropicMaterial.__init__(self,*args,**kwargs)
        self.eps_infinity=0
    
    def epsilon(self,freqs,q=None):
        
        w=freqs*1e-4 #w is inverse microns (why michael??)
        e1 = -73.14*w**2  + 41.59*w - 0.05094
        e2 = 7.122*w**2 - 3.833*w + 0.4894
        
        return e1+1j*e2

PZT=_PZT_()

######
#---Si
######

Si=IsotropicMaterial(eps_infinity=11.7+1.3704e-6j)

############
#---Doped Si
############

try: Si_Doped=DopedSilicon(ne=0,nh=1e18)
except: Logger.exception('Material data not found.',level='warning')

####################
#---SiO2: Lorentzian
####################

SiO2=IsotropicMaterial(eps_infinity=1.9259,\
                        eps_lps=numpy.array([[14.261*1072*49.44,       1072,   49.44],\
                                             [-.18416*1270*215.53,      1270,   215.53],\
                                             [.46239*1205*77.698,       1205,   77.698]]))

##########
#---SiC 4H
##########

omega_p=400
drude_params=[[numpy.sqrt(6.56)*omega_p,omega_p]]*2+\
             [[numpy.sqrt(6.78)*omega_p,omega_p]]
E_TO=796.6; E_LO=971. #E refers to dipole oscillations perpendicular to c-axis (ordinary)
A_TO=782.; A_LO=965. #A refers to dipole oscillations parallel to c-axis (extraordinary)
damping_factor=1
g=5.3584*damping_factor #22 with zmin=1nm, 30nm radius represented data best
eps_inf=[6.56,6.56,6.78]
eps_lps=[[(eps_inf[0]*(E_LO**2-E_TO**2), E_TO, g)]]*2+\
        [[(eps_inf[2]*(A_LO**2-A_TO**2), A_TO, g)]]
SiC_4H=AnisotropicMaterial(eps_infinity=eps_inf,\
                           eps_lps=eps_lps,\
                           drude_params=[])

#SiC_4H=IsotropicMaterial(eps_infinity=eps_inf[2],\
#                           eps_lps=eps_lps[2],\
#                           drude_params=[])
SiC_4H_2=AnisotropicMaterial2(eps_infinity=eps_inf,\
                           eps_lps=eps_lps,\
                           drude_params=[])

SiC_4H_Dispersive=DispersiveAnisotropicMaterial(eps_infinity=eps_inf,\
                                                eps_lps=eps_lps,\
                                                drude_params=[],\
                                                disperse_by=-200,\
                                                lattice_constant=.3e-7)

##########
#---SiC 6H
##########

E_TO=797.; E_LO=969.9 #E refers to propagation along ordinary axes (a or b)
A_TO=788.1; A_LO=965.3 #A refers to propagation along extraordinary axis (c)
damping_factor=1
g_888=2.41e-3*888.7*damping_factor
s_888=4*pi*9.46e-5*888.7**2
g_884=3.11e-3*883.9*damping_factor
s_884=4*pi*1.86e-4*883.9**2
g=12*damping_factor
eps_inf=[6.56,6.56,6.72]
eps_lps=[[(eps_inf[0]*(E_LO**2-E_TO**2), E_TO, g)]]*2+\
        [[(eps_inf[2]*(A_LO**2-A_TO**2), A_TO, g),\
          (s_884,883.9,g_884),\
          (s_888,888.7,g_888)]]
SiC_6H=AnisotropicMaterial(eps_infinity=eps_inf,\
                           eps_lps=eps_lps)

SiC_6H_2=AnisotropicMaterial2(eps_infinity=eps_inf,\
                             eps_lps=eps_lps,\
                             drude_params=[])

#E_TO=795.3 #E refers to propagation along ordinary axes (a or b)
E_TO=797 #These are the literature frequencies
#A_TO=793.2 #A refers to propagation along extraordinary axis (c)
A_TO=788

damping_factor=1

#g_E=4.119*damping_factor
g_E=3.77*damping_factor
#g_A=5.623*damping_factor
g_A=5.82*damping_factor

#s_E=.03188*(8065.544**2)
s_E=.0315*(8065.544**2)
#s_A=.029875*(8065.544**2)
s_A=.0304*(8065.544**2)

#eps_inf=[6.6812,6.6812,6.41]
eps_inf=[6.66, 6.66, 6.368]

eps_lps=[[(s_E, E_TO, g_E)]]*2+\
        [[(s_A, A_TO, g_A)]]
SiC_6H_Ellips=AnisotropicMaterial(eps_infinity=eps_inf,\
                           eps_lps=eps_lps)

##########
#---SiC 3C
##########

E_TO=795.9; E_LO=972.3
eps_inf=6.49
g=10
#omega_p=300
omega_p=0
drude_params=[numpy.sqrt(6.56)*omega_p,omega_p]
eps_lps=[eps_inf*(E_LO**2-E_TO**2),E_TO,g]
SiC_3C=IsotropicMaterial(eps_infinity=eps_inf,\
                         eps_lps=eps_lps,\
                         drude_params=drude_params)

########
#---SiN4
########

SiN4_Bulk=IsotropicMaterial(eps_infinity=2,\
                              eps_lps=numpy.array([[30*840*50,840,50],\
                                                   [30*880*50,880,50],\
                                                   [3*1030*30,1030,30]]))

eps_infinity=1.9259

lp = numpy.array([[14.261,    1072,  49],\
                  [0.18416,  1270,  216],
                  [1.4,    802.3,  80],\
                  [9.972,  462.9, 30],\
                  [0.46239, 1205, 78]])
for i in range(len(lp)): lp[i,0]*=lp[i,1]*lp[i,2]


SiO2_Fei=IsotropicMaterial(eps_infinity=1.9259,\
                                  eps_lps=lp)

####################
#---SiO2: 300nm Greg
####################

SiO2_300nmFei=TabulatedMaterialFromFile('sio2_300nm_extracted_epsilon_cone_A=2a.pickle')

########################
#---SiO2: Amorphous Bulk
########################

##Lorentz-Gauss parameters from Kucirkova & Navratil 1994##
Kucirkova_bulk_params=numpy.array([[.8,1172,13,65],\
                                   [2.9,1090,12,16],\
                                   [2.8,1060,5,26],\
                                   [.4,803,35,27]]).astype(float)
Kucirkova_bulk_params[:,0]*=1e5*numpy.sqrt(pi)/2.
Kucirkova_bulk_params[:,0]/=Kucirkova_bulk_params[:,1] #Divide by frequency
Kucirkova_bulk_params[:,0]/=Kucirkova_bulk_params[:,3] #Divide by gaussian width

SiO2_Bulk=IsotropicMaterial(eps_infinity=1.96,\
                            eps_vps=Kucirkova_bulk_params)

#########################
#---SiO2: Amorphous 300nm
#########################

Kucirkova_300nm_params=numpy.array([[1.73,1147,40,81],\
                                    [1.5,1091,.3,24],\
                                    [3.3,1060,8,30],\
                                    [.35,808,10,32]])
Kucirkova_300nm_params[:,0]*=1e5*numpy.sqrt(pi)/2.
Kucirkova_300nm_params[:,0]/=Kucirkova_300nm_params[:,1] #Divide by frequency
Kucirkova_300nm_params[:,0]/=Kucirkova_300nm_params[:,3] #Divide by gaussian width

SiO2_300nm=IsotropicMaterial(eps_infinity=1.96,\
                             eps_vps=Kucirkova_300nm_params)
SiO2_300nm.name='SiO2_300nm'

###########################
#---SiO2: Amorphous 300nm 2
###########################

Kucirkova_300nm2_params=numpy.array([[1.7,1150,40,80],\
                                    [3.1,1080,.1,26],\
                                    [1.1,1045,10,17],\
                                    [.34,800,.1,56]])
Kucirkova_300nm2_params[:,0]*=1e5*numpy.sqrt(pi)/2.
Kucirkova_300nm2_params[:,0]/=Kucirkova_300nm2_params[:,1] #Divide by frequency
Kucirkova_300nm2_params[:,0]/=Kucirkova_300nm2_params[:,3] #Divide by gaussian width

SiO2_300nm2=IsotropicMaterial(eps_infinity=1.96,\
                              eps_vps=Kucirkova_300nm2_params)

Kucirkova_300nm2_test_params=numpy.array([[1.7*1.1,1150,40,80],\
                                    [3.1*1.1,1080,.1,26],\
                                    [1.1*1.1,1045,10,17],\
                                    [.34*1.1,800,.1,56]])
Kucirkova_300nm2_test_params[:,0]*=1e5*numpy.sqrt(pi)/2.
Kucirkova_300nm2_test_params[:,0]/=Kucirkova_300nm2_test_params[:,1] #Divide by frequency
Kucirkova_300nm2_test_params[:,0]/=Kucirkova_300nm2_test_params[:,3] #Divide by gaussian width

SiO2_300nm2_test=IsotropicMaterial(eps_infinity=1.96,\
                              eps_vps=Kucirkova_300nm2_test_params)

#########################
#---SiO2: Amorphous 144nm
#########################

Kucirkova_144nm_params=numpy.array([[1.26,1156,4,78],\
                                    [3.9,1092,.1,18],\
                                    [4.5,1067,6,35],\
                                    [.28,808,.1,32]])
Kucirkova_144nm_params[:,0]*=1e5*numpy.sqrt(pi)/2.
Kucirkova_144nm_params[:,0]/=Kucirkova_144nm_params[:,1] #Divide by frequency
Kucirkova_144nm_params[:,0]/=Kucirkova_144nm_params[:,3] #Divide by gaussian width

SiO2_144nm=IsotropicMaterial(eps_infinity=1.96,\
                             eps_vps=Kucirkova_144nm_params)

########################################################################################
#--- TaS2: high temperature (metallic) and low temperature (charge-ordered)
########################################################################################

TaS2_NCCDW=TaS2_metal=TabulatedMaterialFromFile('TaS2_eps_230K.csv')
TaS2_CCDW=TabulatedMaterialFromFile('TaS2_eps_30K.csv')

######################
#---Compound Materials
######################

############
#---Graphene
############

SuspendedGraphene=SingleLayerGraphene(chemical_potential=1500,gamma=100)
SupportedGraphene=LayeredMedia(SuspendedGraphene,(SiO2_300nm,300e-7),\
                                 entrance=Air,\
                                 exit=Si)

#############################
#---Bi2Se3: Surface and Films
#############################

#Fermi level of Bi2Se3 can range from -400 to 1600
#Bi2Se3_Surface=TopologicalInsulatorSurface(chemical_potential=1600,gamma=30)
#Bi2Se3_films={'surface':dict([(QLs,LayeredMedia(Bi2Se3_Surface,\
#                                                (Bi2Se3s[QLs],QLs*1e-7),\
#                                                Bi2Se3_Surface,\
#                                                exit=Al2O3_2)) for QLs in [6,15,30,60]]),\
#              'nosurface':dict([(QLs,LayeredMedia((Bi2Se3s[QLs],QLs*1e-7),\
#                                                  exit=Al2O3_2)) for QLs in [6,15,30,60]])}

################################################
#---BSTS: 35nm crystal data from Erik van Heumen
################################################
try:
    BSTS_35nm_Bulk=TabulatedMaterialFromFile(epsfile='Erik_BSTS_epsilon.pickle')
    BSTS_Surface_top=TopologicalInsulatorSurface(chemical_potential=2400,gamma=30)
    BSTS_Surface_bottom=TopologicalInsulatorSurface(chemical_potential=2400,gamma=30)
    BSTS_35nm=LayeredMedia(BSTS_Surface_top,(BSTS_35nm_Bulk,35e-7),BSTS_Surface_bottom,exit=Si)
    BSTS_Bulk=LayeredMedia(BSTS_Surface_top,exit=BSTS_35nm_Bulk)
except IOError:
    Logger.exception('Material data not found.',level='warning')
    
    
    
    
##########################
#---VO2
##########################
#
VO2_Insulating=TabulatedMaterialFromFile('VO2_295K.csv')
VO2_Metallic=TabulatedMaterialFromFile('VO2_360K.csv')

##########################
#---V2O3
##########################
#
V2O3_Insulating=TabulatedMaterialFromFile('Stewart_V2O3film_insulating.csv')
V2O3_Metallic=TabulatedMaterialFromFile('Stewart_V2O3film_metallic.csv')