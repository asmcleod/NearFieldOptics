import os
import numpy
import pickle
from common import misc
from common import numerics
from common.baseclasses import ArrayWithAxes as AWA
from common.log import Logger
from scipy.interpolate import UnivariateSpline

base_dir=os.path.dirname(__file__)

def PtSiTipProfile(zs,L=754.9):
    
    file=open(os.path.join(base_dir,'PtSiTipProfile.csv'))
    zs_loaded,rs_loaded=misc.extract_array(file).astype(float).T #Both in units of a=25nm
    file.close()
    
    zs_loaded*=L/754.9; rs_loaded*=L/754.9 #Expand size by the L specified
    
    dr=rs_loaded[1]-rs_loaded[0]; dz=zs_loaded[1]-zs_loaded[0]
    apex_angle=numpy.arctan2(dr,dz)
    
    z_shift=(1-numpy.sin(apex_angle))
    r_shift=numpy.cos(apex_angle)
    
    zs_loaded+=z_shift
    rs_loaded+=r_shift
    
    zs_added=numpy.arange(500)/500.*z_shift
    rs_added=numpy.sqrt(1-(zs_added-1)**2)
    
    all_rs=numpy.hstack((rs_added,rs_loaded))
    all_zs=numpy.hstack((zs_added,zs_loaded))
    
    interp=UnivariateSpline(x=all_zs,y=all_rs,s=.1)
    rs=interp(zs)

    rs[rs<=1e-6]=1e-6
    rs[zs>L]=1e-6
    
    return rs

