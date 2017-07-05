from numpy import *
from scipy.interpolate import interp1d
from common.baseclasses import AWA

def inversion_wrapper (Beta_inics, Sdata_reals, Sdata_imags, freqs_D, SModel, freqs=None, **kwargs):
    "parameters for the code"
    
    if freqs is None:
        freqs = freqs_D
    wsteps = len(freqs) # steps in between frequencies
    
    dw = (freqs_D[-1]-freqs_D[0])/float(wsteps)
    freq_reverse = kwargs['freq_reverse']
    omega_coeff = kwargs["omega_coeff"]
    omega = (2*pi)/(omega_coeff*dw) # 2 or greater
    zeta = kwargs.pop('zeta') # harmonic oscillator factor
    
    Sdata_array=(Sdata_reals + 1j*Sdata_imags)
    Sdata_function = interp1d(freqs_D,Sdata_array)
    
    from common.numerics import differentiate
    dSdw=differentiate(x=freqs_D,y=Sdata_array)
    ddSdw=differentiate(x=freqs_D,y=dSdw)
        
    dSd1=dSdw.real
    dSd2=dSdw.imag
        
    ddSd1=ddSdw.real
    ddSd2=ddSdw.imag
    
    DSd1 = interp1d(freqs_D,dSd1)
    DSd2 = interp1d(freqs_D,dSd2)
    DdSd1 = interp1d(freqs_D,ddSd1)
    DdSd2 = interp1d(freqs_D,ddSd2)

    derivative = {"DSd1":DSd1,"DSd2":DSd2,"DdSd1":DdSd1,"DdSd2":DdSd2}
    
    kwargs["derivative"] = derivative
    kwargs["Sdata_interp"] = Sdata_function    
        
    def input_derivatives(w, **kwargs):
        derivs = kwargs["derivative"]
        DSd1 = derivs["DSd1"]
        DSd2 = derivs["DSd2"]
        DdSd1 = derivs["DdSd1"]
        DdSd2 = derivs["DdSd2"]
        
        Sdata_func = kwargs["Sdata_interp"] 
        Sdata = Sdata_func(w)
        
        return [Sdata.real,Sdata.imag,DSd1(w),DSd2(w), DdSd1(w),DdSd2(w)]   #Sd1,Sd2, dSd1, dSd2,ddSd1, ddSd2
    
    print("\nStarting the Trace")
    
    def DSigmodel(w,b1,b2,SigModel,**kwargs):
        
        h = .01 #.001*sqrt(b1**2+b2**2) #derivative perhaps can scale with the coordinate values @TODO: does this need to be the same h as earlier?
        b_current = SigModel(w,b1,b2,**kwargs)
        
        "first derivative"
        "x"
        forwardx = SigModel(w,b1 + h, b2,**kwargs)
        backwardx = SigModel(w,b1 - h, b2,**kwargs)
        dx = (forwardx - backwardx)/(2*h)
    
        "y"
        forwardy = SigModel(w,b1, b2 + h,**kwargs)
        backwardy = SigModel(w, b1, b2 - h,**kwargs)
        dy = (forwardy - backwardy)/(2*h)
        
        "second derivatives"
        "x"
        ddx = (forwardx - 2*b_current + backwardx)/(h**2)
        
        "y"
        ddy= (forwardy - 2*b_current + backwardy)/(h**2)
        
        sm = [i for i in range(12)]
        sm[0] = b_current.real  # the model function -- real Sm1
        sm[1] = b_current.imag  # -- imaginary Sm2
        sm[2] = dx.real # x derivative -- real db1Sm1
        sm[3] = dx.imag # -- imaginary db1Sm2
        sm[4] = dy.real #y derivative -- real  db2Sm1
        sm[5] = dy.imag# -- imaginary db2Sm2
        sm[6] = ddx.real# second derivative of x -- real  ddb1Sm1
        sm[7] = ddx.imag# second derivative of x -- imaginary   ddb1Sm2
        sm[8] = ddy.real# second derivative of y -- real  ddb2Sm1
        sm[9] = ddy.imag# second derivative of y -- imaginary   ddb2Sm2
        sm[10] = 0#double.real # dx and dy -- real   db1db2Sm1
        sm[11] = 0#double.imag # dx and dy -- imaginary  db1db2Sm2
        return sm


    "Derivative functions"
    "beta1"
    def ddBeta1(w_i,b1,b2, db1,db2,omega,zeta,**kwargs):
            Sm1, Sm2, db1Sm1, db1Sm2, db2Sm1, db2Sm2, ddb1Sm1,ddb1Sm2,ddb2Sm1,ddb2Sm2,db1db2Sm1,db1db2Sm2  = DSigmodel(w_i,b1,b2,SModel,**kwargs)
            Sd1,Sd2,dSd1, dSd2,ddSd1, ddSd2 = input_derivatives(w_i, **kwargs) 
            num = (omega**2*Sd2*db2Sm1-omega**2*Sm2*db2Sm1+2*zeta*omega*dSd2*db2Sm1 + 
            ddSd2*db2Sm1 - omega**2*Sd1*db2Sm2 + omega**2*Sm1*db2Sm2 -
            2*zeta*omega*dSd1*db2Sm2-ddSd1*db2Sm2+ db2**2*db2Sm2*ddb2Sm1 - 
            db2**2*db2Sm1*ddb2Sm2+2*zeta*omega*db1*db2Sm2*db1Sm1-2*zeta*omega*
            db1*db2Sm1*db1Sm2+2*db1*db2*db2Sm2*db1db2Sm1-2*db1*db2*db2Sm1*db1db2Sm2+
            db1**2*db2Sm2*ddb1Sm1-db1**2*db2Sm1*ddb1Sm2)
            jac=jacobian(db2Sm1,db1Sm2,db2Sm2,db1Sm1,**kwargs)
            if not jac:
                return [0,0] # 0 for we are below jacobian_threshold
            else:
                return [num/jac,1] # 1 because we passed jacobian_threshold
            
    "beta2"
    def ddBeta2(w_i,b1,b2, db1, db2,omega,zeta,**kwargs): 
            Sm1, Sm2, db1Sm1, db1Sm2, db2Sm1, db2Sm2, ddb1Sm1,ddb1Sm2,ddb2Sm1,ddb2Sm2,db1db2Sm1,db1db2Sm2  = DSigmodel(w_i,b1,b2,SModel,**kwargs)
            Sd1,Sd2,dSd1, dSd2,ddSd1, ddSd2 = input_derivatives(w_i, **kwargs) 
            num = (-db1Sm2*(-omega**2*Sd1+omega**2*Sm1-2*zeta*omega*dSd1-
            ddSd1+2*zeta*omega*db2*db2Sm1+db2**2*ddb2Sm1+2*zeta*omega*db1*db1Sm1+
            2*db1*db2*db1db2Sm1+db1**2*ddb1Sm1) + db1Sm1*(-omega**2*Sd2+   
            omega**2*Sm2-2*zeta*omega*dSd2-ddSd2+2*zeta*omega*db2*db2Sm2+
            db2**2*ddb2Sm2+2*zeta*omega*db1*db1Sm2+2*db1*db2*db1db2Sm2+
            db1**2*ddb1Sm2))
            jac=jacobian(db2Sm1,db1Sm2,db2Sm2,db1Sm1,**kwargs)
            if not jac:
                return [0,0] # 0 for we are below jacobian_threshold
            else:
                return [num/jac,1] # 1 because we passed jacobian_threshold
           
    "Jacobian check at dynamic signal values -- called by func1 and ddbeta2"
    def jacobian(db2Sm1,db1Sm2,db2Sm2,db1Sm1,**kwargs):
        #print "db2Sm1",db2Sm1,"db1Sm2",db1Sm2,"db2Sm2:",db2Sm2,"db1Sm1:",db1Sm1
        det = (db2Sm1*db1Sm2-db2Sm2*db1Sm1)
        jacobian_thres = kwargs["jacobian_thres"]
        if (abs(det) < jacobian_thres):
            print "Under Threshold value.\nThe value of the determinate is %s"%det
            return 0
        else: 
            #print "\nThe value of the determinate is %s"%det
            return  det
            
    def heun(x0, t, ddBeta1, ddBeta2,SModel,omega,zeta,**kwargs):
        print "Entering Huen's Method\n"
        n = len( t )
        
        dB1 = []; dB2 = []
        B1 = []; B2 = []
        #hlist = [] 
        db1plist = []; db1clist =[]
        db2plist =[]; db2clist =[]
        
        db1plist.append(0)
        db1clist.append(0)
        db2plist.append(0)
        db2clist.append(0)
        
        sig_list = []
        
        finalBeta = []
        signal_bf = []
        sig_deriv = []
    
        B1.append(x0[0])
        B2.append(x0[1])
        dB1.append(x0[2])
        dB2.append(x0[3])
        
        sig_list.append(SModel(t[0],B1[0], B2[0],**kwargs))

        i = 0
        flag5 = 0
        convergence_counter = 0
        
        iteration_max = kwargs["iteration_max"]
        threshold = kwargs["data_threshold"]
        
        while i<n-1:
            
            Sdata_value=Sdata_function(t[i])
            thres_s = abs((sig_list[-1]-Sdata_value) / Sdata_value)
            print "Best-fit signal value at this position is within %1.2f%% of the data."%(thres_s*100)
            
            print "\nThe current frequency is: %s"%t[i]
            print 'The current position is: beta=%s+i*%s'%(B1[-1],B2[-1])
            print "Signal value in data: %s"%Sdata_value
            print "Signal value from model: %s"%sig_list[-1]
            
            h = t[i+1] - t[i]
            #hlist.append(h)
            
            "Beta 1 Calculation"
            
            b1p = dB1[-1]
            db1p,flag1 = ddBeta1(t[i], B1[-1], B2[-1],dB1[-1],dB2[-1],omega,zeta,**kwargs)
            print 'Beta1 predictor derivative returns: %s  %s'%(db1p,flag1)
            
            if (db1p!=0):db1plist.append(db1p)
            if (flag1 == 0):db1p = db1plist[-1]
            
            b1c =  dB1[-1] + h*db1p
                
            db1c,flag2 = ddBeta1(t[i]+h, B1[-1] + h*b1p,B2[-1], dB1[-1] + h*db1p,dB2[-1],omega,zeta,**kwargs)
            print 'Beta1 corrector derivative returns: %s  %s'%(db1c,flag2)
            
            if (db1c!=0):db1clist.append(db1c)
            if (flag2 == 0):db1c = db1clist[-1] 
            
            "Beta 2 Calculation"
            b2p = dB2[-1]
            db2p,flag3 = ddBeta2(t[i], B1[-1], B2[-1],dB1[-1],dB2[-1],omega,zeta,**kwargs)
            print 'Beta2 predictor derivative returns: %s  %s'%(db2p,flag3)
            
            if (db2p!=0):db2plist.append(db2p)
            if (flag3 == 0):db2p = db2plist[-1] 
                
            b2c =  dB2[-1] + h*db2p
            db2c,flag4 = ddBeta2(t[i]+h,B1[-1], B2[-1] + h*b2p,dB1[-1], dB2[-1] + h*db2p,omega,zeta,**kwargs)
            print 'Beta2 corrector derivative returns: %s  %s'%(db2c,flag4)
            
            if (db2c!=0):db2clist.append(db2c)
            if (flag4 == 0):db2c = db2clist[-1] 

            if (thres_s <= threshold or convergence_counter == iteration_max):
                "Calculation of the i+1 values"
               
                print '\n Recording Values\n %1.2f%% complete'%(100*(t[i]-t[0])/(t[-1]-t[0])), "\n"
                if (flag1 or flag2 == 0): db1p = db1c = 0
                if (flag3 or flag4 == 0): db2p = db2c = 0
                    
                B1.append(B1[-1] + h*(b1p + b1c)/2.0)
                dB1.append(dB1[-1] + h*(db1p + db1c )/2.0)
                
                B2.append(max(B2[-1] + h*(b2p + b2c)/2.0,0)) #Never permit to become negative
                dB2.append(dB2[-1] + h*(db2p + db2c )/2.0)
                sig_list.append(SModel(t[i+1],B1[-1], B2[-1],**kwargs))
                
                if(i==0 and flag5 == 0):
                    finalBeta.append(B1[-1] + 1j*B2[-1])
                    signal_bf.append(SModel(t[i],B1[-1], B2[-1],**kwargs))
                    sig_deriv.append(DSigmodel(i,signal_bf[-1].real,signal_bf[-1].imag,SModel,**kwargs))
                    flag5 = 1
                
                else:
                    finalBeta.append(B1[-1] + 1j*B2[-1])
                    signal_bf.append(SModel(t[i+1],B1[-1], B2[-1],**kwargs))
                    sig_deriv.append(DSigmodel(i+1,signal_bf[-1].real,signal_bf[-1].imag,SModel,**kwargs))
                    i += 1
                convergence_counter = 0 
            
            else:
                if (flag1 or flag2 == 0): db1p = db1c = 0
                if (flag3 or flag4 == 0): db2p = db2c = 0
                
                B1.append(B1[-1] + h*(b1p + b1c)/2.0)
                dB1.append(dB1[-1] + h*(db1p + db1c )/2.0)
                
                B2.append(max(B2[-1] + h*(b2p + b2c)/2.0,0)) #Never permit to become negative
                dB2.append(dB2[-1] + h*(db2p + db2c )/2.0)
                sig_list.append(SModel(t[i+1],B1[-1], B2[-1],**kwargs))
                convergence_counter = convergence_counter + 1
                
                print 'Agreement is insufficient! Beginning convergence iteration %i...'%convergence_counter
              
        sig_deriv = array(sig_deriv)
        finalBeta= array(finalBeta)
        return [finalBeta,signal_bf,sig_deriv]
    
    global Beta_bf,sig_bf,sig_deriv
    print 'Using the following keyword arguments to the signal model:'
    print kwargs
    if(freq_reverse==True):
        freqs = freqs[::-1]
    Beta_bf,sig_bf,sig_deriv = heun(Beta_inics, freqs, ddBeta1, ddBeta2,SModel,omega,zeta,**kwargs)
    
    sig_bf_dxdw = sig_deriv[:,2] + 1j*sig_deriv[:,3]
    sig_bf_dydw = sig_deriv[:,4] + 1j*sig_deriv[:,5]
    
    beta_AWA = AWA(Beta_bf,axes=[freqs],axis_names=['Frequency'])#interp1d(freqs,Beta_bf,bounds_error = False, fill_value = 0)
    sig_AWA = AWA(sig_bf,axes=[freqs],axis_names=['Frequency'])#interp1d(freqs,sig_bf,bounds_error = False, fill_value = 0)
    sig_data_AWA = AWA(Sdata_array, axes=[freqs_D], axis_names=['Frequency'])
    
    best_fit = {"beta_AWA":beta_AWA,"sig_bf_AWA":sig_AWA, "sig_data_AWA":sig_data_AWA,\
                "sig_bf_dxdw":sig_bf_dxdw,"sig_bf_dydw":sig_bf_dydw }

   
    
    return best_fit
 
