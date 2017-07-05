import os
import numpy
import cPickle
from common.baseclasses import ArrayWithAxes as AWA
from common.numerical_recipes import GetQuadrature,GL
from NearFieldOptics.PolarizationModels import azimuthal_charge as az
from matplotlib.pyplot import *

data_dir=os.path.join(os.path.dirname(__file__),'ChargeData')
os.chdir(data_dir)

def view_induced_charge(ds=[2,5,10,20,40,60,80,100,120,140,160],N=500,R=1,L=100,
                            plot=True,**kwargs):

    ds=numpy.array(ds)

    if plot: fig=figure()
    fs=[]
    f_maxes=[]
    for i,d in enumerate(ds):
        
        if i>0: az.reuse_kernel=True
        
        print 'd=%s'%d
        f,eps,e=az.get_charge_dist(N=N,R=R,L=L,**kwargs)
        
        fs.append(f)
        if plot: numpy.log(numpy.abs(f)).plot(lw=2,label=r'$d=%inm$'%d)
        
        max=numpy.max(numpy.abs(f.cslice[:R]))
        f_maxes.append(max)
        
    az.reuse_kernel=False
    
    if plot:
        figure()
        semilogy(ds/float(R),f_maxes)
        xlabel(r'$d/R$'); ylabel(r'$max|\lambda_{ind}|$ [arb. units]')
        
        figure();loglog(ds/float(R),f_maxes)
        xlabel(r'$d/R$'); ylabel(r'$max|\lambda_{ind}|$ [arb. units]')
        
    return fs,f_maxes

def view_exponential_response(qs=10**(numpy.linspace(-3,1,50)),R=1,N=72*4,\
                               quadrature=GL,plot=True,fig=None,**kwargs):
    global all_f_maxes
    global all_fs
    all_f_maxes=[]
    all_fs=[]
    
    #default xmax=inf, xmin=1e-3, good for quasi-infinite object
    zs,ws=GetQuadrature(quadrature=quadrature,N=N)
    
    for i,q in enumerate(qs):
        
        #if i>0: az.reuse_kernel=True
        
        print 'q-vector [1/R]:',q
        exp_decay=R/float(q)
        az.exp_decay=exp_decay
        
        fs,f_maxes=view_induced_charge(ds=[0],R=R,L=3*(zs/q).max(),N=N,plot=False,V_ext='exponential',\
                                       quadrature=(zs/q,ws/q),**kwargs)
        all_fs.append(fs[0])
        all_f_maxes.append(f_maxes[0])
        
    az.reuse_kernel=False
        
    z_axis=all_fs[0].axes[0]/float(R); z_axis-=z_axis.min()
    
    all_f_maxes=AWA(all_f_maxes,\
                    axes=[qs],\
                    axis_names=[r'$q-vector\,\left[1/a\right]$'])
    all_fs=AWA(numpy.array(all_fs).astype(float),\
               axes=[qs,zs],\
               axis_names=[r'$q-vector\,\left[1/a\right]$','z x q'])
        
    if plot:
        if fig: figure(fig.number)
        else: figure()
        semilogx(qs,all_f_maxes,'-o',lw=1.5)
        ylabel(r'$max|\lambda_{induced}|\,\left[arb.\right]$',fontsize=17)
        title('Induced Charge v. $q$-vector',fontsize=17)
        legend(loc='best')
        grid()
        
    return all_f_maxes,all_fs

#Deprecated
def analyze_responses(tapers=[0,10,20,30,45],\
                     qs=10**(numpy.linspace(-3,1,50)),\
                     R=1,\
                     length_ratios=[10,25,50,100,150,200],\
                     N=2000):
    
    global all_responses
    global all_fs
    
    for i,taper in enumerate(tapers):
        all_responses={}; all_fs={}
        
        file_path=os.path.join(data_dir,\
                               'responses_and_charge_dists_taper=%s_lengthratios%sto%s_withbessel.pickle'%\
                               (taper,min(length_ratios),max(length_ratios)))
        file=open(file_path,'w')
        
        for j,length_ratio in enumerate(length_ratios):
            L=R*length_ratio
            
            print '----- Progress: currently on %i of %i -----'%(i*len(length_ratios)+j+1,\
                                                    len(tapers)*len(length_ratios))
            print '----- Taper angle: %s -----'%taper
            print '----- Length ratio: %s -----'%length_ratio
            print '----- R=%s, L=%s, N=%s -----'%(R,L,N)
            
            responses,fs=view_exponential_response(qs=qs,taper_angle=taper,\
                                                   L=L,N=N,R=R,\
                                                   smoothing=1,\
                                                   plot=False)
            all_responses[length_ratio]=responses
            all_fs[length_ratio]=fs
        
        print '----- Saving data -----'
        
        cPickle.dump(dict(responses=all_responses,charge_dists=all_fs),file)
        file.close()
        print '----- Done saving -----'
        
def fit_inf_charges_triexp(file='responses_and_charge_dists_taper=15_infLength.pickle',\
                           interp_Nqs=100,params0=[-1,5,.01,.01,-.05],\
                           sequential=True,high_q_first=False,max_error=.2,\
                           xtol=1e-8,ftol=1e-8):
    
    from common.numerical_recipes import ParameterFit
    from matplotlib import pyplot
    
    fs=cPickle.load(open(file))['fs'].astype(numpy.float)
    qs,zs=fs.axes
    
    #Get initial fit with triple exponential#
    def tripleExp(zs,args):
        
        a,alpha,b,beta,c=args
        return a*numpy.exp(-alpha*zs)+\
               b*numpy.exp(-beta*zs)+\
               c*numpy.exp(+c/(a/alpha+b/beta)*zs)
    
    #Start with the first q-value#
    if high_q_first: f0=fs.cslice[qs.max()]
    else: f0=fs.cslice[qs.min()]
    fit0=ParameterFit(zs,f0,tripleExp,params0,xtol=xtol,ftol=ftol)[0]
    fmodel=tripleExp(zs,fit0)
    error=numpy.sqrt(numpy.sum(numpy.abs(fmodel-f0)**2)/numpy.sum(numpy.abs(f0)**2))
    pyplot.figure()
    pyplot.plot(zs,f0,lw=2,label='Actual')
    pyplot.plot(zs,fmodel,lw=2,ls='--',label='Model')
    pyplot.gca().set_xscale('log'); pyplot.legend(loc='best');pyplot.show()
    
    a0,alpha0,b0,beta0,c0=fit0
    gamma0=-c0/(a0/alpha0+b0/beta0)
    
    print 'a0=%s, b0=%s, c0=%s'%(a0,b0,c0)
    print 'alpha0=%s, beta0=%s, gamma0=%s'%(alpha0,beta0,gamma0)
    proceed=raw_input('Error=%1.2e, is the initial fit good? [y]/n:  '%error)
    if proceed.lower().startswith('n'):
        print 'Try different `params0` to attain an initial best-fit.'
        return
    
    #Define a new function#
    #Initial values will have all=1
    def asymptExp(zs,args):
        
        A,alpha,B,beta,C=args
        gamma=-c0*C/(a0/(alpha0*alpha)+\
                         b0*B/(beta0*beta))\
                    /gamma0
        
        #First exponential will be the strongest term,
        #overall magnitude coefficient A
        return a0*A*(numpy.exp(-alpha0*alpha*zs)+\
                     b0/a0*B*numpy.exp(-beta0*beta*zs)+\
                     c0/a0*C*numpy.exp(-gamma0*gamma*zs))
    
    #With good fit, interpolate fs#
    interp_qs=numpy.logspace(numpy.log(qs[0])/numpy.log(10),\
                             numpy.log(qs[-1])/numpy.log(10),\
                             interp_Nqs)
    fs=fs.interpolate_axis(interp_qs,axis=0)
    qs=fs.axes[0]
    if qs[0]<qs[-1] and high_q_first or\
       qs[0]>qs[-1] and not high_q_first:
        fs=fs[::-1];qs=fs.axes[0]
    
    #Fit them all
    fits=[]; errors=[]; starting_fit=[1,1,1,1,1]
    for i,f in enumerate(fs):
        print 'Fitting charge q=%1.2e, %i of %i...'%(qs[i],\
                                                     i+1,len(fs))
        
        fit=ParameterFit(zs,f,asymptExp,starting_fit,xtol=xtol,ftol=ftol)[0]
        fits.append(fit)
        
        fmodel=asymptExp(zs,fit)
        error=numpy.sqrt(numpy.sum(numpy.abs(fmodel-f)**2)/numpy.sum(numpy.abs(f)**2))
        print 'Current error: %1.2e.'%error
        errors.append(error)
        if error>max_error:
            print 'Maximum tolerable error exceeded!'
            break
        
        if sequential: starting_fit=fit
        if i==len(fs)-1:
            pyplot.figure(); title('q*a=%s'%(qs[-1]))
            pyplot.plot(zs,f,lw=2,label='Actual')
            pyplot.plot(zs,fmodel,lw=2,ls='--',label='Model')
            pyplot.gca().set_xscale('log'); pyplot.legend(loc='best');pyplot.show()
    
    fits=numpy.array(fits).transpose()
    A,alpha,B,beta,C=fits
    gamma=-c0*C/(a0/(alpha0*alpha)+\
                 b0*B/(beta0*beta))\
                /gamma0
    global params
    params=dict(A=A,alpha=alpha,B=B,beta=beta,C=C,gamma=gamma)
    qs_axis=qs[:len(A)]
    for key,param in params.iteritems():
        params[key]=AWA(param,axes=[qs_axis],axis_names=['q [1/R]'])
        
    pyplot.figure()
    pyplot.gca().set_xscale('log'); pyplot.gca().set_yscale('log')
    pyplot.title('Coefficients')
    for key in ['A','B','C']:
        params[key].plot(lw=2,label=key)
    pyplot.figure()
    pyplot.gca().set_xscale('log'); pyplot.gca().set_yscale('log')
    pyplot.title('Decay Constants')
    for key in ['alpha','beta','gamma']:
        params[key].plot(lw=2,ls='--',label=key)
        
    params0=dict(a0=a0,alpha0=alpha0,b0=b0,beta0=beta0,c0=c0,gamma0=gamma0)
    params.update(params0); params['errors']=AWA(errors,axes=[qs_axis],axis_names=['q [1/R]'])
        
    return params

def fit_inf_charges_biexp(file='responses_and_charge_dists_taper=15_infLength.pickle',\
                           interp_Nqs=100,params0=[-1,5,.01],\
                           sequential=True,high_q_first=False,max_error=.2,\
                           xtol=1e-8,ftol=1e-8):
    
    from common.numerical_recipes import ParameterFit
    from matplotlib import pyplot
    
    fs=cPickle.load(open(file))['fs'].astype(numpy.float)
    qs,zs=fs.axes
    
    #Get initial fit with triple exponential#
    def biExp(zs,args):
        
        a,alpha,b=args
        return a*numpy.exp(-alpha*zs)+\
               b*numpy.exp(+b/(a/alpha)*zs)
    
    #Start with the first q-value#
    if high_q_first: f0=fs.cslice[qs.max()]
    else: f0=fs.cslice[qs.min()]
    fit0=ParameterFit(zs,f0,biExp,params0,xtol=xtol,ftol=ftol)[0]
    fmodel=biExp(zs,fit0)
    error=numpy.sqrt(numpy.sum(numpy.abs(fmodel-f0)**2)/numpy.sum(numpy.abs(f0)**2))
    pyplot.figure()
    pyplot.plot(zs,f0,lw=2,label='Actual')
    pyplot.plot(zs,fmodel,lw=2,ls='--',label='Model')
    pyplot.gca().set_xscale('log'); pyplot.legend(loc='best');pyplot.show()
    
    a0,alpha0,b0=fit0
    beta0=b0/(a0/alpha0)
    
    print 'a0=%s, b0=%s'%(a0,b0)
    print 'alpha0=%s, beta0=%s'%(alpha0,beta0)
    proceed=raw_input('Error=%1.2e, is the initial fit good? [y]/n:  '%error)
    if proceed.lower().startswith('n'):
        print 'Try different `params0` to attain an initial best-fit.'
        return
    
    #Define a new function#
    #Initial values will have all=1
    def asymptExp(zs,args):
        
        A,alpha,B=args
        beta=-b0*B/(a0/(alpha0*alpha))\
                    /beta0
        
        #First exponential will be the strongest term,
        #overall magnitude coefficient A
        return a0*A*(numpy.exp(-alpha0*alpha*zs)+\
                     b0/a0*B*numpy.exp(-beta0*beta*zs))
    
    #With good fit, interpolate fs#
    interp_qs=numpy.logspace(numpy.log(qs[0])/numpy.log(10),\
                             numpy.log(qs[-1])/numpy.log(10),\
                             interp_Nqs)
    fs=fs.interpolate_axis(interp_qs,axis=0)
    qs=fs.axes[0]
    if qs[0]<qs[-1] and high_q_first or\
       qs[0]>qs[-1] and not high_q_first:
        fs=fs[::-1];qs=fs.axes[0]
    
    #Fit them all
    fits=[]; errors=[]; starting_fit=[1,1,1]
    for i,f in enumerate(fs):
        print 'Fitting charge q=%1.2e, %i of %i...'%(qs[i],\
                                                     i+1,len(fs))
        
        fit=ParameterFit(zs,f,asymptExp,starting_fit,xtol=xtol,ftol=ftol)[0]
        fits.append(fit)
        
        fmodel=asymptExp(zs,fit)
        error=numpy.sqrt(numpy.sum(numpy.abs(fmodel-f)**2)/numpy.sum(numpy.abs(f)**2))
        print 'Current error: %1.2e.'%error
        errors.append(error)
        if error>max_error:
            print 'Maximum tolerable error exceeded!'
            break
        
        if sequential: starting_fit=fit
        if i==len(fs)-1:
            pyplot.figure(); title('q*a=%s'%(qs[-1]))
            pyplot.plot(zs,f,lw=2,label='Actual')
            pyplot.plot(zs,fmodel,lw=2,ls='--',label='Model')
            pyplot.gca().set_xscale('log'); pyplot.legend(loc='best');pyplot.show()
    
    fits=numpy.array(fits).transpose()
    A,alpha,B=fits
    beta=-b0*B/(a0/(alpha0*alpha))\
                /beta0
    global params
    params=dict(A=A,alpha=alpha,B=B,beta=beta)
    qs_axis=qs[:len(A)]
    for key,param in params.iteritems():
        params[key]=AWA(param,axes=[qs_axis],axis_names=['q [1/R]'])
        
    pyplot.figure()
    pyplot.gca().set_xscale('log'); pyplot.gca().set_yscale('log')
    pyplot.title('Coefficients')
    for key in ['A','B']:
        params[key].plot(lw=2,label=key)
    pyplot.figure()
    pyplot.gca().set_xscale('log'); pyplot.gca().set_yscale('log')
    pyplot.title('Decay Constants')
    for key in ['alpha','beta']:
        params[key].plot(lw=2,ls='--',label=key)
        
    params0=dict(a0=a0,alpha0=alpha0,b0=b0,beta0=beta0)
    params.update(params0); params['errors']=AWA(errors,axes=[qs_axis],axis_names=['q [1/R]'])
        
    return params