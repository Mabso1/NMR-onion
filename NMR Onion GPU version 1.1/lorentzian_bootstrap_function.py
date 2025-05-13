#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:53:23 2022

@author: mathies
"""

import pandas as pd
import numpy as np
import sys
import torch
from torch.special import expit
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
path="/home/mathies/Desktop/nmr onion/NMR onion final"
sys.path.append(path)
from model_prediction_fuctions import *
from torch_loss_fn_full_freq_lor import *
from onion_filter_experimental import *
from helper_functions import *


def lorentzian_wild_boostrap(B,CI_level,alpha,omega,scale1,scale2,t,k,y,fs,model_name,low_ppm,high_ppm,freq,SF,O1p):
    Boots=[]
    y_dem,fs_new,tn_new,t_new,freq_new,hz_moved=signal_decimate(low_ppm=low_ppm,high_ppm=high_ppm, y_filt=y, freq=freq, SF=SF, O1p=O1p, fs=fs)
   # par_hat=np.concatenate(np.array([alpha,omega+hz_moved,scale1,scale2])) # collect the parameters
    par_hat = np.concatenate([
    np.atleast_1d(alpha),
    np.atleast_1d(omega + hz_moved),
    np.atleast_1d(scale1),
    np.atleast_1d(scale2)]) # collect the parameters
    
    
    y_hat=pred(theta=par_hat,m=k,t=t,y=y, model_name=model_name)
    resids=y_hat-y
    for l in range(0,B):
        y_resamp=wild_bootstrap(y_hat=y_hat, residuals=resids)
        
        y_dem,fs_new,tn_new,t_new,freq_new,hz_moved=signal_decimate(low_ppm=low_ppm,high_ppm=high_ppm, y_filt=y_resamp, freq=freq, SF=SF, O1p=O1p, fs=fs)
        
        
        wild_fit=fit_parameter_error(par_hat=par_hat, fs=fs, t=t_new, k=k, y=y_dem)
        
        par_res=par_est(fit=wild_fit, k=k, model_name=model_name)
        omega_boot=par_res['omega_est']-hz_moved
        alpha_boot=expit_sci(par_res['alpha_est']) # convert to expit scale
        scale1_boot=par_res['scale'] # uncomment if model is skewed 
        scale2_boot=par_res['scale1'] # uncomment if model is skewd
        
        B_temp=np.concatenate((alpha_boot,omega_boot,scale1_boot,scale2_boot))
            # append samples
        Boots.append(B_temp)
        print(l) # print sample number
        
    boot_res=pd.DataFrame(Boots) # store bootstap samples
    if (boot_res.isnull().values.any()==True):
        boot_res=rm_nans(boot_res) # remove nans

    
  # collect bootstap samples from each parameter
    alpha_boot_clean=boot_res.T[0:k] 
    omega_boot_clean=boot_res.T[k:2*k]
    scale1_boot_clean=boot_res.T[2*k:(2*k+1)]
    scale2_boot_clean=boot_res.T[(2*k+1):(3*k+1)]
    
    # remove outliers based on IQR*1.5
    Q1=omega_boot_clean.quantile(0.25,axis=1)
    Q3=omega_boot_clean.quantile(0.75,axis=1)
    IQR=Q3-Q1
    
    no_outlier = omega_boot_clean.T[~((omega_boot_clean.T < (Q1 - 1.5 * IQR)) |(omega_boot_clean.T > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    row_idx=no_outlier.index.values
    
    alpha_boot_clean=alpha_boot_clean.T.loc[row_idx].T
    omega_boot_clean=omega_boot_clean.T.loc[row_idx].T
    scale1_boot_clean=scale1_boot_clean.T.loc[row_idx].T
    scale2_boot_clean=scale2_boot_clean.T.loc[row_idx].T
    
    # get Confidence intervals for each parameter
    alpha=1-CI_level # get CI level
    
    alpha_CI_upper=alpha_boot_clean.T.quantile(1-alpha/2)
    alpha_CI_lower=alpha_boot_clean.T.quantile(alpha/2)
    
    omega_CI_upper=omega_boot_clean.T.quantile(1-alpha/2)
    omega_CI_lower=omega_boot_clean.T.quantile(alpha/2)
    
    
    scale1_CI_upper=scale1_boot_clean.T.quantile(1-alpha/2)
    scale1_CI_lower=scale1_boot_clean.T.quantile(alpha/2)
    
    scale2_CI_upper=scale2_boot_clean.T.quantile(1-alpha/2)
    scale2_CI_lower=scale2_boot_clean.T.quantile(alpha/2)
    
    # get standard error of each parameter
    
    alpha_std=alpha_boot_clean.T.std()

    omega_std=omega_boot_clean.T.std()
        
    scale1_std=scale1_boot_clean.T.std()

    scale2_std=scale2_boot_clean.T.std()
    
    
    # output:
    # bootrap samples:
     #1= bootsamples alpha
     #2= bootsamples omega
     #3= bootsamples eta
     #4=bootsamples scale1
     #5= bootsamples scale2
   # confidence interval based off bootstrap samples:
     #1= bootsamples alpha CI
     #2= bootsamples omega CI
     #3= bootsamples eta CI
     #4=bootsamples scale1 CI
     #5= bootsamples scale2 CI
   # std error based off bootstrap samples:
     #1= bootsamples alpha std
     #2= bootsamples omega std
     #3= bootsamples eta std
     #4=bootsamples scale1 std
     #5= bootsamples scale2 std
     
    names = ['alpha_boot','omega_boot','scale1_boot','scale2_boot',\
             'alpha_CI_lower','alpha_CI_upper',\
             'omega_CI_lower','omega_CI_upper',\
             'scale1_CI_lower','scale1_CI_upper',\
             'scale2_CI_lower','scale2_CI_upper',\
              'alpha_std','omega_std','scale1_std','scale2_std'] # collect parameters for output
    out=(alpha_boot_clean,omega_boot_clean,scale1_boot_clean,scale2_boot_clean,\
         alpha_CI_lower,alpha_CI_upper,\
         omega_CI_lower,omega_CI_upper,\
         scale1_CI_lower,scale1_CI_upper,\
         scale2_CI_lower,scale2_CI_upper,\
         alpha_std,omega_std,scale1_std,scale2_std)
    named_out = dict(zip(names, out))
    
    return named_out


def wild_bootstrap(y_hat, residuals):
    """Produces resampled fitted response variable with the provided residuals."""
    bs_residuals =  residuals*np.random.normal(size=len(residuals))
    return y_hat + bs_residuals

def fit_parameter_error(par_hat,fs,t,k,y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SP = torch.nn.Softplus()
    parm=[]
    N=len(t)
    
    p=Variable(torch.from_numpy(par_hat),requires_grad=True).type(torch.FloatTensor).to(device) # convert to tensor
    t_in=torch.from_numpy(t) # convert to tensor
    y_in=torch.from_numpy(y) # add random noise with loss value at the k'th sinusoid as varaince and 0 as mean
    train_data=TensorDataset(t_in,y_in)
    train_load=DataLoader(train_data,batch_size=N,shuffle=False,num_workers=1)
    
    # generate the model of time domain voigt
    class sinusoid(torch.nn.Module):
            def __init__(self):
                
                # define number of paameters a =decay, f=frequency, scale=scaling constant 1 , scale2=scaling constant 2
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(p[0:k],dtype=float))
                self.f = torch.nn.Parameter(torch.tensor(p[k:2*k],dtype=float))
                self.scale = torch.nn.Parameter(torch.tensor(p[2*k:2*k+1],dtype=float))
                self.scale2 = torch.nn.Parameter(torch.tensor(p[2*k+1:3*k+1],dtype=float))

                
            def forward(self, t_in,y_in,k):
                # signal pole matrix with decay
                 Z= torch.exp(torch.outer(t_in,1j * 2 * torch.pi * self.f-SP(self.scale)*torch.exp(1j*self.scale2)*expit(self.a)))

                 # small diagnonal to aviod instability
                 diag_small=torch.eye(k*1)*(1j+1)*10**-6
                 # complex amplitudes
             #    L = torch.linalg.cholesky(torch.conj(Z).T@Z+diag_small)

              #   A=torch.cholesky_solve(torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1)), L,upper=False)
                 
                 A=torch.linalg.solve(torch.conj(Z).T@Z+diag_small.to(device),torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1))) 
                 # prediction of the model
                 pred=Z@A
            
                 return pred,A # return predicions and complex amplitudes 
        
    model = sinusoid().to(device)
   
    def closure():
      if torch.is_grad_enabled():
          optimizer.zero_grad()
      y_pred = model(t_batch,y_batch,k)
      loss = loss_fn(y_pred, y_batch,k)           
      if loss.requires_grad:
          loss.backward()
      #    print('loss:', loss.item())
      return loss
    
    optimizer=torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=100, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-09, history_size=10, line_search_fn="strong_wolfe")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
     # make the forward pass via a loop
    epochs=1 # full batch so only one epoch needed 
    for epoch in range(epochs):
        running_loss=0.0
        for steps, (t_batch,y_batch) in enumerate(train_load): 
             t_batch,y_batch=t_batch.to(device),y_batch.to(device)
          
        # Update weights
           #  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0,norm_type=2)
            # with warmup_scheduler.dampening():
             
             optimizer.step(closure)
        loss=closure()
        running_loss+=loss.item()
        scheduler.step()
        print(f"Epoch:{epoch+1:02}/epochs Loss: {running_loss:.5e}")
    
   
    model_out=list(model.parameters())
    alpha1=(model_out[0]).cpu().detach().numpy()
    omega1=model_out[1].cpu().detach().numpy()
    scale1=model_out[2].cpu().detach().numpy()
    scale2=model_out[3].cpu().detach().numpy()
    
    # parameter vector
    parms_k=np.concatenate((alpha1,omega1,Softplus(scale1),scale2))
    
    parm.append(parms_k) # collect parameters per compontent
    
    names = ['par'] # collect parameters, residuals, number of componts, BIC and AIC for output
    out=(parm[0:k])
    named_out = dict(zip(names, out))
    
    return(named_out)

#%%
#lorentzian_wild_boostrap(2, 0.95, alpha, omega, scale1, scale2, hz_moved, t_new, k, y_dem/np.linalg.norm(y_dem), "skewed_lorentzian")





