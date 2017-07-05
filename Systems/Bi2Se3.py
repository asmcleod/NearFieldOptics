from numpy import *
from matplotlib.pyplot import *
from NearFieldOptics import Materials as mat
from NearFieldOptics import TipModels as tip

def plot_rps(mus=linspace(100,3000,50),gamma=10,QLs=6,\
             freqs=linspace(400,1400,400), qs=linspace(.001,6,400)*1/(25e-7)):

    media=mat.LayeredMedia(mat.Bi2Se3_Surface,\
                           (mat.Bi2Se3_Bulk,QLs*1e-7),\
                            mat.Bi2Se3_Surface,\
                           exit=mat.Al2O3)
    rps={}
    media.get_layers()[0].gamma=gamma
    media.get_layers()[2].gamma=gamma

    for i,mu in enumerate(mus):
        
        print('Working on mu=%i...'%mu)
        media.get_layers()[0].chemical_potential=mu
        media.get_layers()[2].chemical_potential=mu
        
        rp=media.reflection_p(freqs,qs);rp.set_axes(axes=[None,qs*25e-7])
        figure(10);abs(rp).transpose().plot(aspect='auto')
        
        subplots_adjust(bottom=.12,right=.75)
        clim(0,8)
        figure(10).axes[1].set_ylabel(r'$|\beta(\omega,q)|$',fontsize=25,rotation=270)
        grid()
        
        xlabel('$q\,[a^{-1}]$',fontsize=23)
        ylabel(r'$\omega\,[cm^{-1}]$',fontsize=25)
        text(3.5,450,r'$\mu=%icm^{-1}$'%mu+'\n'+r'$\Gamma=%icm^{-1}$'%gamma,fontsize=20,bbox={'boxstyle':'round','facecolor':'white'})
        title(r'$%iQL\,Bi_2Se_3$ on $Al_2O_3$'%QLs,fontsize=20)
        
        savefig('rp_%iQL_gamma=%s_%03d.png'%(QLs,gamma,i),dpi=150)
        print 'Done.'
        rps[mu]=rp
        clf()
        subplots_adjust(left=.125,right=.9,bottom=.1,top=.9,wspace=.2,hspace=.2)
        
    return rps

def test():
    
    figure();plot([1,2,3]);show(); raw_input('proceed?')
    
    figure();plot([4,5,6]);show()