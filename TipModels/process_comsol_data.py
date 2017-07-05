import numpy
import os
from common.log import Logger
from common.misc import extract_array
from common.baseclasses import AWA
from common import numerics
from matplotlib.pyplot import *
from scipy.interpolate import RectBivariateSpline
from NearFieldOptics.PolarizationModels import azimuthal_charge as az
from common import plotting

def test():
    
    figure();plot(numpy.arange(5))#;draw()
    
    proceed=raw_input('Is displayed tip profile OK? [y]/n: ')
    if proceed.lower().startswith('n'):
        raise ValueError("Taper=%s and a=%s must be wrong!"%(taper,a))

def ExtractChargeDistribution(filename,taper=20,plane='xz',\
                              pct_radius_offset=2,reverse_z=True,\
                              reload_fields=True,a=20,geometry='cone',\
                              comment_char='%',**kwargs):
    
    global fields1_arr,fields2_arr,radii,charges,pref,zs_wo_offset,zs_offset,rsi,zsi
    
    if reload_fields:
        try: del fields1_arr,fields2_arr,charges
        except NameError: pass
        
        ##Load from file##
        Logger.write('Loading field data...')
        
        if plane=='xz':
            xs,ys,zs,Enorm,Exr,Exi,Eyr,Eyi,Ezr,Ezi=extract_array(open(filename),\
                                                                 comment_char=comment_char).astype(float).T
            fields1=[Exr,Exi]
            fields2=[Ezr,Ezi]
            pos1=xs*1e9
            
        elif plane=='yz':
            xs,ys,zs,Enorm,Exr,Exi,Eyr,Eyi,Ezr,Ezi=extract_array(open(filename),\
                                                                 comment_char=comment_char).astype(float).T
            fields1=[Eyr,Eyi]
            fields2=[Ezr,Ezi]
            pos1=ys*1e9
            
        elif plane=='rz':
            rs,zs,Enorm,Ephi,Err,Eri,Ezr,Ezi=extract_array(open(filename),\
                                                      comment_char=comment_char).astype(float).T
            fields1=[Err,Eri]
            fields2=[Ezr,Ezi]
            pos1=rs*1e9
            
        else:
            raise ValueError("Don't understand plane %s!"%plane)
        
        pos2=zs*1e9
        
        ##We want to make radial values positive - flip about mirror axis
        if (pos1<=0).all():
            pos1*=-1
            fields1[0]*=-1
            fields1[1]*=-1
        pos1-=pos1.min() #start radial coordinate at zero
        Logger.write('\tDone.')
        
        ##Build arrays out of data##
        Logger.write('\tBuilding arrays out of fields...')
        fields1_arr=numerics.array_from_points(zip(pos1,pos2), fields1[0]+1j*fields1[1])
        fields2_arr=numerics.array_from_points(zip(pos1,pos2), fields2[0]+1j*fields2[1])
        Enorm=numerics.array_from_points(zip(pos1,pos2), Enorm)
        
        rs,zs=fields1_arr.axes
        if reverse_z:
            zs=zs.max()-zs
            fields1_arr.set_axes(axes=[rs,zs]); fields1_arr=fields1_arr.sort_by_axes()
            fields2_arr.set_axes(axes=[rs,zs]); fields2_arr=fields2_arr.sort_by_axes()
        Logger.write('\tDone.')
    
    Eperpr=numpy.sqrt(fields1_arr.real**2+fields2_arr.real**2)*numpy.sign(fields1_arr.real)
    Eperpi=numpy.sqrt(fields1_arr.imag**2+fields2_arr.imag**2)*numpy.sign(fields1_arr.imag)
    
    ##Get axes and update to reflect z-offset##
    figure()
    numpy.sqrt(Eperpr**2+Eperpi**2).plot(log_scale=False,plotter=contourf)
    gca().set_aspect('equal')
    draw()
    rs,zs=fields1_arr.axes
    
    proceed='n'
    while proceed.lower().startswith('n'):
        z_offset=raw_input('Enter z-offset of the tip apex (default=0): ')
        if not z_offset: z_offset=0
        else: z_offset=float(z_offset)
        
        zs_wo_offset=zs[zs>=z_offset]
        zs_offset=zs_wo_offset-zs_wo_offset.min()
        
        ##Get radii and confirm profile##
        if not kwargs.has_key('L'): kwargs['L']=zs_offset.max()
        radii=az.get_radii(zs_offset,z0=0,R=a,geometry=geometry,**kwargs)
        
        plot(radii*(1+.01*pct_radius_offset),zs_wo_offset,color='r',lw=2)
        draw()
        proceed=raw_input('Is displayed tip profile OK? [y]/n: ')
        
    drdz=numerics.differentiate(x=zs_offset,y=radii)
    
    ##Create interpolators and get values##
    Logger.write('\tExtracting field values along tip surface...')
    rsi,zsi=radii*(1+.01*pct_radius_offset),zs_wo_offset
    Interpr=RectBivariateSpline(rs,zs,Eperpr)
    Interpi=RectBivariateSpline(rs,zs,Eperpi)
    Eperp=Interpr.ev(rsi,zsi)+1j*Interpi.ev(rsi,zsi)
    
    dAdz=2*numpy.pi*(radii/numpy.float(a))*numpy.sqrt(1+drdz**2)
    charges=AWA(dAdz*Eperp/(4*numpy.pi),axes=[zs_offset/numpy.float(a)],axis_names=['Z [a]'])
    
    return charges
    