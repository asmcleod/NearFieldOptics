import numpy
from common.baseclasses import ArrayWithAxes as AWA
from NearFieldOptics import materials as mat
from NearFieldOptics.PolarizationModels import tip_models as tip

from common.log import Logger
Logger.setFilterDepth(1)

Efs=numpy.linspace(200,1200,300)
gammas=[15,30,50,100,200,300]

def get_s2s(Efs=Efs,gammas=gammas,Nqs=1000,model=tip.ExtendedMonopoleModel,zmin=7,amplitude=50,a=25.89):

	s2_d={}
	for gamma in gammas:
		mat.SuspendedGraphene.gamma=gamma
    		s2s=numpy.array([model(1192,rp=mat.SuspendedGraphene.reflection_p,chemical_potential=Ef,normalize_to=mat.Si.reflection_p,zmin=zmin,amplitude=amplitude,a=a,harmonic=2,Nqs=1000) for Ef in Efs])
		s2_d['%i S2'%gamma]=numpy.abs(s2s)
		s2_d['%i P2'%gamma]=numpy.arctan2(s2s.imag,s2s.real)

	s2_d['Efs']=Efs
	return s2_d
