import os
import numpy as np
from common import numerics as num
from common import numerical_recipes as numrec
from common.baseclasses import AWA
from matplotlib import pyplot as plt

base_dir=os.path.dirname(__file__)
source_pts={'fwd':np.loadtxt(os.path.join(base_dir,'sourcefwd.txt'),delimiter=','),\
            'bwd':np.loadtxt(os.path.join(base_dir,'sourcebwd.txt'),delimiter=',')}
destination_pts={'fwd':np.loadtxt(os.path.join(base_dir,'destinationfwd.txt'),delimiter=','),\
                 'bwd':np.loadtxt(os.path.join(base_dir,'destinationbwd.txt'),delimiter=',')}

def smooth_grid(source_pts,dest_pts,deg=2,
                dy=.8,dx=1,roty=0,rotx=-5,\
                plot=True):
    
    xs,ys=zip(*source_pts)
    xs_dest,ys_dest=zip(*dest_pts)
    
    #Convenience function
    def poly_smoothed(row_xs,row_ys,deg=3):
        row_xs=np.array(row_xs)
        p=np.polyfit(row_xs,row_ys,deg)
        new_ys=np.sum([p[i]*row_xs**(deg-i) for i in range(deg+1)],axis=0)
        return new_ys
    
    #For adjusting y coordinates
    test_xs,test_ys=zip(*[num.rotate_vector((x,y),roty) for x,y in zip(xs,ys)])
    test_ys,xs,ys,xs_dest,ys_dest=zip(*sorted(zip(test_ys,xs,ys,xs_dest,ys_dest)))
    
    rows=[]
    for i in range(len(ys)):
        x,y=xs[i],ys[i]
        test_dy=test_ys[i]-test_ys[i-1]
        if i==0: new_row=[(x,y)]
        elif test_dy>dy:
            rows.append(new_row)
            new_row=[(x,y)]
        else: new_row.append((x,y))
    rows.append(new_row)
    
    if plot:
        plt.figure(); plt.title('Did we identify rows correctly?')
        for row in rows:
            these_xs,these_ys=zip(*row)
            plt.plot(these_xs,these_ys,marker='o',ls='')
    
    smoothed_ys=[]
    for row in rows:
        row_xs,row_ys=zip(*row)
        new_ys=poly_smoothed(row_xs,row_ys,deg=deg)
        smoothed_ys+=list(new_ys)
    
    #For adjusting x coordinates
    test_xs,test_ys=zip(*[num.rotate_vector((x,y),rotx) for x,y in zip(xs,smoothed_ys)])
    test_xs,xs,ys,smoothed_ys,xs_dest,ys_dest=zip(*sorted(zip(test_xs,xs,ys,smoothed_ys,xs_dest,ys_dest)))
    
    rows=[]
    for i in range(len(xs)):
        x,y=xs[i],smoothed_ys[i]
        test_dx=test_xs[i]-test_xs[i-1]
        if i==0: new_row=[(x,y)]
        elif test_dx>dx:
            rows.append(new_row)
            new_row=[(x,y)]
        else: new_row.append((x,y))
    rows.append(new_row)
    
    if plot:
        plt.figure(); plt.title('Did we identify columns correctly?')
        for row in rows:
            these_xs,these_ys=zip(*row)
            plt.plot(these_xs,these_ys,marker='o',ls='')
    
    smoothed_xs=[]
    for row in rows:
        row_xs,row_ys=zip(*row)
        new_xs=poly_smoothed(row_ys,row_xs,deg=deg)
        smoothed_xs+=list(new_xs)
        
    return [list(zip(smoothed_xs,smoothed_ys)),\
            list(zip(xs_dest,ys_dest))]

default_rotation=0
default_center=[0,0]

default_out_xs=np.linspace(-5,60,650)
default_out_ys=np.linspace(-5,55,600)

def getXYGrids(shape, size, rotation=None, center=None):
    """Generate a meshgrid and rotate it by RotRad radians.
    Adapted from:
        https://stackoverflow.com/questions/29708840/rotate-meshgrid-with-numpy"""

    # Clockwise, 2D rotation matrix
    if rotation is None: rotation=default_rotation
    angleRad=np.pi*rotation/180.
    RotMatrix = np.array([[np.cos(angleRad),  np.sin(angleRad)],
                          [-np.sin(angleRad), np.cos(angleRad)]])

    Nx,Ny=shape
    dx,dy=size
    xs=np.linspace(-dx/2.,dx/2.,Nx)
    ys=np.linspace(-dy/2.,dy/2.,Ny)

    Xs,Ys = np.meshgrid(xs, ys)
    Xs,Ys = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([Xs, Ys]))
    
    if center is None: center=default_center
    x0,y0=center
    Xs+=x0
    Ys+=y0
    
    return Xs,Ys

grids=None

def UndistortImage(image,image_size,\
                   image_rotation=None,image_center=None,\
                    out_xs=None,out_ys=None,\
                    direction='fwd',regenerate_grids=True,\
                    **kwargs):
    """Remember the recipe for fixin gwyddion image orientation:  `image0=image0.T[:,::-1]`"""
    
    global grids
    
    if out_xs is None: out_xs=default_out_xs
    if out_ys is None: out_ys=default_out_ys
    
    if grids is None or regenerate_grids:
        s=source_pts[direction]; d=destination_pts[direction]
        grids=numrec.AffineGridsFromFeaturePoints(d,[s],xs=out_xs,ys=out_ys)
    
    in_Xs,in_Ys=getXYGrids(image.shape,image_size,\
                           rotation=image_rotation,center=image_center)
    undistorted=numrec.InterpolateImageToAffineGrid(image,grid_pts=grids['grid_pts'][0],\
                                                    image_xgrid=in_Xs,image_ygrid=in_Ys,
                                                    **kwargs)
    
    return AWA(undistorted,axes=[out_xs,out_ys])