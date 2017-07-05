import os
import cPickle
import numpy
from matplotlib.pyplot import *
from common import numerics as num
from common.log import Logger
from common.baseclasses import ArrayWithAxes as AWA
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat

database_dir=os.path.join(os.path.dirname(__file__),'InversionDatabase')
database_title_template='NearFieldValuesOnRpGrid.pickle'

def MakeNearFieldGrid(Nrps=200,rp_re_lims=[1e-3,50],rp_im_lims=[1e-3,25],\
                      freq=1000,a=30,amplitude=80,
                      Nqs=72,normalization='Au',**kwargs):
    
    tip.LRM.load_params['freq']=freq*a*1e-7
    
    if normalization=='Au': normalize_to=mat.Au
    elif normalization=='Si': normalize_to=mat.Si
    else: normalize_to=getattr(mat,normalization)
    
    ###Set up rp values and frequency values to run through##
    Nrps=Nrps+Nrps%2 #round up to nearest 2-multiple
    x=numpy.logspace(numpy.log(rp_re_lims[0])/numpy.log(10),\
                     numpy.log(rp_re_lims[1])/numpy.log(10),\
                     Nrps/2)
    x=numpy.hstack((-x[::-1],x))
    #x=numpy.linspace(-rp_re_lims[1],rp_re_lims[1],Nrps)
    y=numpy.logspace(numpy.log(rp_im_lims[0])/numpy.log(10),\
                     numpy.log(rp_im_lims[1])/numpy.log(10),\
                     Nrps)
    #y=numpy.linspace(rp_im_lims[0],rp_im_lims[1],Nrps)
    #Give axes proper shape for broadcasting
    x=numpy.reshape(x,(len(x),1)); y=numpy.reshape(y,(1,len(y)))
    rp_vals=x+y*1j
    
    ###Get normalization value###
    Logger.write('Computing normalization signal...')
    tip.verbose=False #Turn off messages
    norm_signal=tip.LightningRodModel(freq,rp=getattr(normalize_to,'reflection_p'),\
                                      a=a,amplitude=amplitude,\
                                      Nqs=Nqs,**kwargs)
    tip.LRM.load_params['reload_model']=False
    
    ###Begin to compute the near-field values###
    signal2_vals=numpy.zeros(rp_vals.shape,dtype=numpy.complex)
    signal3_vals=numpy.zeros(rp_vals.shape,dtype=numpy.complex)
    
    #Make some bogus frequency axis centered at frequency
    bogus_freq_axis=numpy.linspace(freq-1e-4,freq+1e-4,Nrps)
    
    #Compute array of near-field signals for a particular Re[rp] and all possible Im[rp]
    for i in range(Nrps):
        Logger.write('\tComputing set of near-field values %i of %i - %1.2f%% complete.'%(i,Nrps,\
                                                                                          i/float(Nrps)*100))
        rp_segment=AWA(rp_vals[i],axes=[bogus_freq_axis],axis_names=['Frequency'])
        signal=tip.LightningRodModel(bogus_freq_axis,rp=rp_segment,\
                                    a=a,amplitude=amplitude,\
                                    Nqs=Nqs,**kwargs)
        signal2_vals[i,:]=signal['signal_2']/norm_signal['signal_2']
        signal3_vals[i,:]=signal['signal_3']/norm_signal['signal_3']
                                        
    signal2_vals=AWA(signal2_vals,axes=[x,y],\
                    axis_names=['Re[rp]','Im[rp]'])
    signal3_vals=AWA(signal3_vals,axes=[x,y],\
                    axis_names=['Re[rp]','Im[rp]'])
    signal_vals=dict(signal_2=signal2_vals,signal_3=signal3_vals)
    
    tip.LRM.load_params['reload_model']=True
    tip.verbose=True
    
    ###Construct the dictionary description of this data###
    d=dict(amplitude=amplitude,normalization=normalization)
    d.update(tip.LRM.geometric_params)
    d.update(tip.LRM.quadrature_params)
    d.update(tip.LRM.load_params)
    d.pop('comsol_filename')
            
    ###Get previous database if it exists###
    database_path=os.path.join(database_dir,database_title_template)
    if os.path.exists(database_path):
        Logger.write('Getting previous database...')
        file=open(database_path,'r')
        try:
            database=cPickle.load(file)
            
            ##Search for identical entries in database##
            ds=[pair[0] for pair in database]; found_match=False
            for i,this_d in enumerate(ds):
                matches=[this_d.has_key(key) and this_d[key]==val for key,val in d.iteritems()]
                #If we find a match in this d#
                if not False in matches:
                    Logger.write('Removing an identical pre-existing database entry...')
                    index=[pair[0] for pair in database].index(this_d)
                    database.pop(index)
                    
        except: file.close(); database=[]
        
    else: database=[]
    
    ###Add to database###
    database.append((d,signal_vals))
    Logger.write('Writing to database...')
    file=open(database_path,'w'); cPickle.dump(database,file); file.close()
    Logger.write('Done.')
    
_cached_databases_={}
    
def ExtractRpFromDatabase(signal_vals,harmonic=2,\
                          freq=1000,a=30,amplitude=80,
                          normalization='Au',\
                          starting_rp=0.1,direction=1,max_dx=5,max_dy=5,\
                          desired_accuracy=.005,n_iter=10,\
                          interpolation='linear',\
                          figures=True,\
                          cache=True):
    
    tip.LRM.load_params['freq']=freq*a*1e-7
    
    ###Construct the dictionary description of this data###
    ignore_keys=['b','x0','comsol_filename','reload_model','interpolation','Nqs','Nzs']
    d=dict(amplitude=amplitude,normalization=normalization)
    d.update(tip.LRM.geometric_params)
    d.update(tip.LRM.quadrature_params)
    d.update(tip.LRM.load_params)
    for key in ignore_keys: d.pop(key)
    
            
    ###Open database###
    global _cached_databases_
    if cache and _cached_databases_:
        database=_cached_databases_ #Try to access name, if we cannot we need to load db
    else:
        database_path=os.path.join(database_dir,database_title_template)
        Logger.raiseException('Database file "%s" does not exist!  Build a database of signal values first.'%database_path+\
                              '  Or, change tip parameters (e.g. geometry) to coincide with an existing database.',\
                              unless=os.path.exists(database_path), exception=OSError)
        file=open(database_path,'r'); database=cPickle.load(file); file.close()
        if cache: _cached_databases_=database
        
    ###Scan database###
    ds=zip(*database)[0]; found_match=False
    for i,this_d in enumerate(ds):
        matches=[this_d.has_key(key) and this_d[key]==val for key,val in d.iteritems()]
        if not False in matches:
            found_match=True; break
    if not found_match: print d.keys()[matches.index(False)]
    Logger.raiseException('There does not exist a database entry with your choice of settings:\n'+\
                          '%s\n'%d+'Try building one first!',\
                          unless=found_match, exception=ValueError)
    Logger.write('Using database entry:\n'+\
                 '\n'.join(['%s:\t%s'%(key,value) for key,value in d.iteritems()]))
    signal_vals_db=database[i][1]['signal_%i'%harmonic]
    
    ##Reverse inversion direction if desired##
    if direction==-1: signal_vals=signal_vals[::-1]
    freqs=signal_vals.axes[0]
    
    ###Try to triangulate each value in *signals* on the database of signal values###
    global best_signals,best_rps
    best_signals=[]
    best_rps=[]
    latest_rp=starting_rp
    for i,signal_val in enumerate(signal_vals):
        Logger.write('\tInverting signal value %i of %i at f=%1.2f cm^-1 - %1.2f%% complete.'%(i+1,len(signal_vals),freq,\
                                                                                               i/float(len(signal_vals))*100))
        
        ##Start loop to converge on this frequency slice##
        #It's good to do this interpolation hard-core
        slice_db=signal_vals_db
        xs_db,ys_db=slice_db.axes
        #Interpolate to within 5 of starting radius
        min_x=numpy.max((xs_db.min(),numpy.real(latest_rp)-max_dx))
        max_x=numpy.min((xs_db.max(),numpy.real(latest_rp)+max_dx))
        min_y=numpy.max((ys_db.min(),numpy.imag(latest_rp)-max_dy))
        max_y=numpy.min((ys_db.max(),numpy.imag(latest_rp)+max_dy))
        subset_db=slice_db.interpolate_axis(numpy.linspace(min_x,max_x,10),axis=0,\
                                                 kind=interpolation)\
                           .interpolate_axis(numpy.linspace(min_y,max_y,10),axis=1,\
                                                 kind=interpolation)
        j=1
        
        error=1
        while True:
            #Logger.write('\tIteration %i...'%j)
            x_grid_db,y_grid_db=subset_db.axis_grids
            diffs=numpy.sqrt(numpy.abs((signal_val-subset_db)/signal_val)**2 \
                             +numpy.abs((latest_rp-(x_grid_db+y_grid_db*1j))/latest_rp)**2)
            best_x,best_y=diffs.locate(diffs.min())[0]
            best_signal=subset_db.cslice[best_x,best_y]
            best_rp=best_x+best_y*1j
            error=numpy.abs((signal_val-best_signal)/signal_val)
            
            if j==n_iter or error<desired_accuracy:
                latest_rp=best_rp
                Logger.write('\tAchieved relative error after %i iterations:\t%1.2f%%'%(j,error*100))
                best_rps.append(best_rp)
                best_signals.append(best_signal)
                break
                
            xs_db,ys_db=subset_db.axes
            dx=numpy.max(numpy.diff(xs_db)); dy=numpy.max(numpy.diff(ys_db))
            min_x=numpy.max((xs_db.min(),best_x-dx)); max_x=numpy.min((xs_db.max(),best_x+dx))
            min_y=numpy.max((ys_db.min(),best_y-dy)); max_y=numpy.min((ys_db.max(),best_y+dy))
            subset_db=slice_db.interpolate_axis(numpy.linspace(min_x,max_x,10),axis=0,\
                                                 kind=interpolation)\
                               .interpolate_axis(numpy.linspace(min_y,max_y,10),axis=1,\
                                                 kind=interpolation)
            j+=1
        
    best_rps=AWA(best_rps,axes=[freqs],axis_names=['Frequency [cm^-1]'])
    best_signals=AWA(best_signals,axes=[freqs],axis_names=['Frequency [cm^-1]'])
    
    avg_error=numpy.sqrt(numpy.mean(numpy.abs(best_signals-signal_vals)**2/abs(signal_vals)**2))*100
    Logger.write('Average error in best-fit signal values:\t%1.2f%%'%avg_error)
    
    if figures:
        figure()
        numpy.abs(signal_vals_db).plot(log_scale=True,plotter=contourf)
        plot(best_rps.real,best_rps.imag,marker='+',color='c')
        
    return {'beta':best_rps,'signal':best_signals,'error':avg_error,'epsilon':(1+best_rps)/(1-best_rps)}

