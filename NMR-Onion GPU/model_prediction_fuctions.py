#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:31:27 2022

@author: mathies
"""
import numpy as np
from scipy.special import expit as expit_sci

# get paramters out as a vector based on best model and its estimated parameters
def par_hat_output(best_model,par_res):
    if (best_model=="skewed_lorentzian"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        scale1=par_res['scale'] # 
        scale2=par_res['scale1']#
        
        #print(np.array([scale2]))
      #  print((alpha,omega,scale1,scale2))
        pars_hat=np.concatenate((alpha,omega,scale1,scale2)) #
        
    elif (best_model=="skewed_pvoigt"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        eta=par_res['eta_est'] 
        scale1=par_res['scale'] 
        scale2=par_res['scale1']
        pars_hat=np.concatenate((alpha,omega,eta,scale1,scale2)) #
        
    elif (best_model=="skewed_genvoigt"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        eta=par_res['eta_est'] 
        scale1=par_res['scale'] 
        scale2=par_res['scale1']
        pars_hat=np.concatenate((alpha,omega,eta,scale1,scale2)) #
        
    elif (best_model=="genvoigt"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        eta=par_res['eta_est'] 
        scale1=par_res['scale'] 
        pars_hat=np.concatenate((alpha,omega,eta,scale1)) #   
        
    elif (best_model=="pvoigt"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        eta=par_res['eta_est'] 
        scale1=par_res['scale'] 
        pars_hat=np.concatenate((alpha,omega,eta,scale1)) #    
    
    elif (best_model=="lorentzian"):
        omega=par_res['omega_est']
        alpha=par_res['alpha_est']
        scale1=par_res['scale'] 
        pars_hat=np.concatenate((alpha,omega,scale1)) #   
    
    return(pars_hat)

# automatical mode selection based of either BIC or AIC 
def model_auto_select(model_compare_BIC,model_compare_AIC,fits,omega_hz_filtered,BIC):
    model_names=np.array(["skewed_lorentzian","skewed_pvoigt","skewed_genvoigt","lortentzian","pvoigt","genvoigt"])
    model_compare_BIC[np.isnan(model_compare_BIC)]=0.0
    model_compare_AIC[np.isnan(model_compare_AIC)]=0.0
    
    if (BIC==True):
        idx_model=np.argmin(model_compare_BIC)
    else:
        idx_model=np.argmin(model_compare_AIC)
    
    model=model_names[idx_model]
    
    best_fit=fits[idx_model]
    
    k=len(omega_hz_filtered)
    
    par_results=par_est(fit=best_fit, k=k, model_name=model)
      
    return par_results,model,k

###softplus function ###
# defines the softplus activation function for numpy implementation
### inputs ###
# x= paramter value in the SP domain, to avoid overflow
def Softplus(x): 
    # Reference: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# prediction functions for evaulation of results
#extract estimated parameters from fitting algorithmn
def par_est(fit,k,model_name):
        
    if(model_name=="skewed_pvoigt" or model_name=="skewed_genvoigt"):  
        a_est=(np.hstack(fit['par'])[0:k])
        f_est=(np.hstack(fit['par'])[k:(2*k)])
        e_est=(np.hstack(fit['par'])[(2*k):(3*k)])
        scale_est=(np.hstack(fit['par'])[(3*k):(3*k+1)])
        scale_est_2=(np.hstack(fit['par'])[(3*k+1):(4*k+1)])
              
        idx=np.argsort(f_est)
        f_est=np.stack(f_est)
        a_est=np.stack(a_est)
        e_est=np.stack(e_est)
        scale_est_2=np.stack(scale_est_2)
           
        f_est=f_est[idx]
        a_est=a_est[idx]
        e_est=e_est[idx]
        scale_est_2=scale_est_2[idx]
        
        names = ['alpha_est', 'omega_est','eta_est','scale','scale1']
        out=(a_est,f_est,e_est,scale_est,scale_est_2)
        named_out = dict(zip(names, out))
        
        return named_out
    
    elif(model_name=="pvoigt" or model_name=="genvoigt"):
        a_est=(np.hstack(fit['par'])[0:k])
        f_est=(np.hstack(fit['par'])[k:(2*k)])
        e_est=(np.hstack(fit['par'])[(2*k):(3*k)])
        scale_est=(np.hstack(fit['par'])[(3*k):(3*k+1)])
        
        idx=np.argsort(f_est)
        f_est=np.stack(f_est)
        a_est=np.stack(a_est)
        e_est=np.stack(e_est)
           
        f_est=f_est[idx]
        a_est=a_est[idx]
        e_est=e_est[idx]
        
        
        names = ['alpha_est', 'omega_est','eta_est','scale']
        out=(a_est,f_est,e_est,scale_est)
        named_out = dict(zip(names, out))
        
        return named_out
        
    elif(model_name=="skewed_lorentzian"):
        a_est=(np.hstack(fit['par'])[0:k]) # decay rate
        f_est=(np.hstack(fit['par'])[k:(2*k)]) # frequnecy 
        scale_est=(np.hstack(fit['par'])[(2*k):(2*k+1)]) # scaling constant 1
        scale_est_2=(np.hstack(fit['par'])[(2*k+1):(3*k+1)]) # scaling constant 2
    
        idx=np.argsort(f_est)
        f_est=np.stack(f_est)
        a_est=np.stack(a_est)
        scale_est_2=np.stack(scale_est_2)
        
        f_est=f_est[idx]
        a_est=a_est[idx]
        scale_est_2=scale_est_2[idx]
        
        names = ['alpha_est', 'omega_est','scale','scale1']
        out=(a_est,f_est,scale_est,scale_est_2)
        named_out = dict(zip(names, out))
        
        return named_out
    
    elif(model_name=="lorentzian"):
        a_est=(np.hstack(fit['par'])[0:k]) # decay rate
        f_est=(np.hstack(fit['par'])[k:(2*k)]) # frequnecy 

        idx=np.argsort(f_est)
        f_est=np.stack(f_est)
        a_est=np.stack(a_est)
       
        f_est=f_est[idx]
        a_est=a_est[idx]

        names = ['alpha_est', 'omega_est']
        out=(a_est,f_est,scale_est,scale_est_2)
        named_out = dict(zip(names, out))
        
        return named_out
    
    else:
        return print("please enter model_name as: lortentzian, skewed_lorentzian,pvoigt,skewed_pvoigt,genvoigt or skewed_genvoigt")        
        
   


### inputs ###
# theta=estimated parameters (alpha,omega,eta,scale_constant1,scale_constant2)
# t=time in seconds
# y=Fid (can be normalzied or unnormalized)
# model_name= name of the model should be one of the following: lortentzian,skewed_lorentzian,pvoigt,skewed_pvoigt,genvoigt or skewed_genvoigt
# retuns amplitudes (A) and model matrix (Z) seperately 
def preds_split(theta,t,m,y,model_name):
     
     if (model_name=="skewed_genvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))# pole matrix
         decay=np.exp(np.power(np.outer(t,np.repeat(1,m)),Softplus(theta[2 * m:3 * m]))*-theta[3 * m:3 * m+1]*np.exp(1j*theta[3 * m+1:4 * m+1])*expit_sci(theta[0 * m:1 * m]))
         Z=Z_poles*(decay)
         
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y)
         
         out=A,Z
         
         return out
     
     elif(model_name=="genvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))#
         decay=np.exp(np.power(np.outer(t,np.repeat(1,m)),Softplus(theta[2 * m:3 * m]))*-theta[3 * m:3 * m+1]*expit_sci(theta[0 * m:1 * m]))
         Z=Z_poles*(decay)
         
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         out=A,Z
         
         return out
         
     elif(model_name=="skewed_pvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))
   
         decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))  

         Z=Z_poles*(decay)
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         out=A,Z
         
         return out
     
     elif(model_name=="pvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))
   
         decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*expit_sci(theta[0 * m:1 * m])))  

         Z=Z_poles*(decay)
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         out=A,Z
         
         return out
     
     elif(model_name=="skewed_lorentzian"):
         Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*theta[2 * m+1:3 * m+1])*expit_sci(theta[0:m]))))
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         out=A,Z
         
         return out
         
     elif(model_name=="lorentzian"):
         Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-Softplus(theta[0:m]))))
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         out=A,Z
         
         return out
         
     else:
         return print("please enter model_name as: lortentzian, skewed_lorentzian,pvoigt,skewed_pvoigt,genvoigt or skewed_genvoigt")
     


# retuns a collect of the underlying single sinusoids of a fit 
def single_sinusoid(theta,t,m,y,model_name):
    N=len(y)
    splits=preds_split(theta,t,m,y,model_name)
    sinusoid_collect=[]
    for i in range(0,m):
        ZZ=splits[1][:,i]
        AA=splits[0][i]
        single_sin=np.array([AA]).T@ZZ.reshape(1,N)
        sinusoid_collect.append(single_sin)
    return sinusoid_collect

# ### preds function ###
# defines the prediction function for the fitting algorithm
### inputs##
# theta= vector of parameters (alpha,omega,eta)=(decay,frequency,weigthing constant)
# t= times series increments
# m= number of components in the selected model
# y= time series data (FID)
# model_name=model_name= name of the model should be one of the following: lortentzian,skewed_lorentzian,pvoigt,skewed_pvoigt,genvoigt or skewed_genvoigt
### outputs ##
# return predicted model in the time domian (FID)
def pred(theta,t,m,y,model_name):
     
     if (model_name=="skewed_genvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))# pole matrix
         decay=np.exp(np.power(np.outer(t,np.repeat(1,m)),Softplus(theta[2 * m:3 * m]))*-theta[3 * m:3 * m+1]*np.exp(1j*theta[3 * m+1:4 * m+1])*expit_sci(theta[0 * m:1 * m]))
         Z=Z_poles*(decay)
         
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         preds=Z@A # fid prediction
         
         return preds
     
     elif(model_name=="genvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))#
         decay=np.exp(np.power(np.outer(t,np.repeat(1,m)),Softplus(theta[2 * m:3 * m]))*-theta[3 * m:3 * m+1]*expit_sci(theta[0 * m:1 * m]))
         Z=Z_poles*(decay)
         
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y)
         
         preds=Z@A # fid prediction
         
         return preds
         
     elif(model_name=="skewed_pvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))#-Softplus(theta[0 * m:1 * m])))
   
         decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))  

         Z=Z_poles*(decay)
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         preds=Z@A # fid prediction
         
         return preds
     
     elif(model_name=="pvoigt"):
         Z_poles = np.exp(np.outer(t, 1j * 2 * np.pi * theta[1 * m:2 * m]))
   
         decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*expit_sci(theta[0 * m:1 * m])))  

         Z=Z_poles*(decay)
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         preds=Z@A # fid prediction
         
         return preds
     
     elif(model_name=="skewed_lorentzian"):
         # signal pole matrix with decay
         Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*theta[2 * m+1:3 * m+1])*expit_sci(theta[0:m]))))
    
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y)  

         preds=Z@A
         
         return preds
         
     elif(model_name=="lorentzian"):
         Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-Softplus(theta[0:m]))))
     
         A=np.linalg.solve(np.conj(Z).T@Z,np.conj(Z).T@y) 
         
         preds=Z@A # fid prediction
         
         return preds
         
     else:
         return print("please enter model_name as: lortentzian, skewed_lorentzian,pvoigt,skewed_pvoigt,genvoigt or skewed_genvoigt")   