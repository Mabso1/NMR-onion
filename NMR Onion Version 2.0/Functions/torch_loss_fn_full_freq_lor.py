#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:47:14 2022

@author: mathies
"""
import torch
import numpy as np
from scipy.special import expit as expit_sci
from torch.special import expit
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import warnings
from joblib.externals.loky.backend.context import get_context



warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # disable DeprecationWarning dewarnings
warnings.filterwarnings("ignore", category=UserWarning)#UserWarning disable (pytorch numpy conversion "warnings")

###softplus function ###
# defines the softplus activation function for numpy implementation
### inputs ###
# x= paramter value in the SP domain, to avoid overflow
def Softplus(x): 
    # Reference: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

### preds function ###
# defines the prediction function for the fitting algorithm
### inputs##
# theta= vector of parameters (alpha,omega,eta)=(decay,frequency,weigthing constant)
# t= times series increments
# m= number of components in y_t=sum^{k}_{i}=(A_i*exp(2j*pi*omega_i*t+phi_i)*exp(-alpha_i*t)
# y= time series data (FID)
### outputs ##
# return predicted model in the time domian (FID)
def obj_nodelay(theta,t,m,y):

  # signal pole matrix
    Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*expit_sci(theta[2 * m+1:2 * m+2])*np.pi/4)*Softplus(theta[0:m]))))
     
     # small diagnonal to aviod instability
    diag_small=np.identity(1*m)*(1j+1)*10**-6
     # complex amplitudes
    A=np.linalg.solve(np.conj(Z).T@Z+diag_small,np.conj(Z).T@y) 
     # prediction of the model
    model=Z@A
       
    diff = y - model # residuals
    
    func=np.real(np.dot(np.conj(diff),diff)) # real dot product of conjugated residuals and residuals
    
    phases = np.arctan2(np.imag(A),np.real(A))
    mu = np.sum(phases) / m
    func += np.sum((phases - mu) ** 2) / (np.pi*m)
    
    return func



def grid_search_alpha(omega_hz_filtered,y,t,k):

    omega_guess=omega_hz_filtered
    scale1_guess=np.array([1.0])
    scale2_guess=np.array([0.0])
    alpha_space=np.linspace(1,20,401)
    
    alpha_list=[]
    obj_list=[]
    for i in range(0,len(alpha_space)):
        alpha_guess=np.repeat(alpha_space[i],omega_guess.shape)
        
        pars=np.concatenate([alpha_guess,omega_guess,scale1_guess,scale2_guess])
        
        obj_value=obj_nodelay(theta=pars, t=t, m=k, y=y)
    
        alpha_list.append(alpha_guess)
        obj_list.append(obj_value)
    
    min_idx=np.argmin(obj_list)
    
    alpha_min=alpha_list[min_idx]
    
    return alpha_min[0]


### preds function ###
# defines the prediction function for the fitting algorithm
### inputs##
# theta= vector of parameters (alpha,omega,eta)=(decay,frequency,weigthing constant)
# t= times series increments
# m= number of components in y_t=sum^{k}_{i}=(A_i*exp(2j*pi*omega_i*t+phi_i)*exp(-alpha_i*t)
# y= time series data (FID)
### outputs ##
# return predicted model in the time domian (FID)
def preds_nodelay(theta,t,m,y):
     # signal pole matrix with decay
     #Z_poles = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m])))#-Softplus(theta[0:m]))))
     Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*expit_sci(theta[2 * m+1:3 * m+1])*np.pi/4)*expit_sci(theta[0:m]))))
     
     # small diagnonal to aviod instability
     diag_small=np.identity(1*m)*(1j+1)*10**-6
     # complex amplitudes
     A=np.linalg.solve(np.conj(Z).T@Z+diag_small,np.conj(Z).T@y) 
     # prediction of the model
     model=Z@A
     
     return model

### pytorch obj function ###
# defines the objective function for the fitting algorithm
### inputs ###
# pred= vector of predicted parameters (alpha,omega,eta,scale1,scale2)=(decay,frequency,weigthing constant,scaling constant1, scaling constant 2)
# y_in= time series data (FID) stored as a tensor
### outputs ###
# returns loss value with phase penalty as resid@resid**H, H=conjugated , with resid=residuals
def loss_fn(pred,y_in,k):
    resid= torch.reshape(y_in,(len(y_in),1)) - pred[0] # residuals
    resid=resid.flatten()
    
    obj=torch.real(torch.dot(torch.conj(resid),resid)) # real dot product of conjugated residuals and residuals
    
    phases = torch.arctan2(torch.imag(pred[1]),torch.real(pred[1])) # get phases from arctan2(Im(amplitude),Re(amplitude))
    mu = torch.sum(phases.flatten()) / k # mean phase
    obj += torch.sum((phases.flatten() - mu) ** 2) / (torch.pi*k) # add phases variance as penalty term to the objective func 
    
    return obj

### k_map_AIC ###
# defines the Akaike information criterion (AIC) for model selection
# AIC is less strict than BIC
### inputs ###
# resi= residuals of fitted model with k number of components
# k= number of components
# N= number of data points
### outputs###
# returns AIC for the k'th number of components, lowest value indicates which model to choose 
def k_map_AIC(resi,k,N):
    SSE=np.real(np.dot(np.conj(resi),resi)) # estimated sum of squared error term from residuals of model k
    MSE=SSE/(N) # estimated mean squared error from SSE of model k
    loglik_temp=-N/2*( (1+np.log(2*np.pi))+np.log(MSE/N) ) # gaussian likelihood
    K_map_temp=-2*(loglik_temp)+(2*3*k+1) # AIC formula= -2*log_lik/N+2*p/N, p=number of parameters (3 ekstra added per compontent, plus 2 addtional for the scaling constants) 
   
    return(K_map_temp)


### k_map_BIC ###
# defines the Bayesian information criterion (BIC) for model selection
# BIC is more strict than AIC
### inputs ###
# resi= residuals of fitted model with k number of components
# k= number of components
# N= number of data points
### outputs###
# returns BIC for the k'th number of components, lowest value indicates which model to choose 
def k_map_BIC(resi,k,N):
    SSE=np.real(np.dot(np.conj(resi),resi)) # estimated sum of squared error term from residuals of model k
    MSE=SSE/(N) # estimated mean squared error from SSE of model k
    loglik_temp=-N/2*( (1+np.log(2*np.pi))+np.log(MSE/N) ) # gaussian likelihood
    K_map_temp=-2*(loglik_temp)+(1+2*3*k*np.log(N)) # BIC formula= -2*log_lik/N+(2*p*log(N))/N, p=number of parameters (3 ekstra added per compontent, plus 2 addtional for the scaling constants)
   
    return(K_map_temp)


### flex_decay_fitting_onion_nodelay_refine_lor ###
# defines the sequential fitting algorithmn of NMR-onion used to deconvolute an NMR FID
### inputs ###
# omega= a list of detected peaks found by the detection algorithmn 
# tn= discrete time series
# t= time series
# y= digitally filtered data filtered data (preferably normalized) 
### outputs ###
# returns a list containing peak numbers, parameters, AIC and BIC 
def onion_lor(omega,tn,t,fs,y):
    SP = torch.nn.Softplus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(y)
    k = len(omega)

    # Prepare initial parameters
    init = np.hstack((
        np.zeros(k),         # alpha
        np.array(omega),     # omega
        np.zeros(k),         # eta
        [10.0],              # scale (single value)
        np.zeros(k)          # scale2z
    ))

    t_in = torch.tensor(t, dtype=torch.float32, device=device)
    y_in = torch.tensor(y, dtype=torch.complex64, device=device)

    p = torch.tensor(init, dtype=torch.float64)
    
    # generate the pytorch model class
    class sinusoid_refit(torch.nn.Module):
            def __init__(self, p, k):
                
                # define number of paameters a =decay, f=frequency, scale=scaling constant 1 , scale2=scaling constant 2
                super().__init__()
                self.a = torch.nn.Parameter(p[0:k].clone().to(torch.float64))
                self.f = torch.nn.Parameter(p[k:2 * k].clone().to(torch.float64))
                self.scale = torch.nn.Parameter(p[2 * k:2 * k + 1].clone().to(torch.float64))
                self.scale2 = torch.nn.Parameter(p[2 * k + 1:3 * k + 1].clone().to(torch.float64))
                
            def forward(self, t,y,k):
                # signal pole matrix with decay
                 Z= torch.exp(torch.outer(t_in,1j * 2 * torch.pi * self.f-SP(self.scale)*torch.exp(1j*expit(self.scale2)*torch.pi/4)*expit(self.a)))
                # Z_real = torch.clamp(Z.real, min=-80, max=80)
                 #Z_imag = torch.clamp(Z.imag, min=-80, max=80)
                 
                 #Z = Z_real + 1j * Z_imag
                 #print(Z)
                 # small diagnonal to aviod instability
                 diag_small=(torch.eye(k*1)*(1j+1)*10**-6).to(device)
                 # complex amplitudes
               #  L = torch.linalg.cholesky(torch.conj(Z).T@Z+diag_small)
                 Z = Z.to(torch.complex64)
                 y = y.to(torch.complex64)
                # A=torch.cholesky_solve(torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1)), L,upper=False)
                 
                 A=torch.linalg.solve(torch.conj(Z).T@Z+diag_small,torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1))) 
                 # prediction of the model
                 pred=Z@A
            
                 return pred,A # return predicions and complex amplitudes 
        
    model = sinusoid_refit(p,k).to(device) # define model
    
    #y_pred = model(t_in,y_in,k)
    #loss_current = loss_fn(y_pred, y_in,k)
   # print(loss_current)
    
    # call the LBFGS optimizer from pytorch         
    optimizer=torch.optim.LBFGS(model.parameters(), lr=0.9, max_iter=200, tolerance_grad=1e-5, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    
   # Define the early stopping criteria
    best_loss = float("inf")
    patience, counter = 2, 0

    def closure():
        optimizer.zero_grad()
        y_pred = model(t_in, y_in, k)
        loss = loss_fn(y_pred, y_in, k)

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ NaN or Inf loss detected in closure. Skipping step.")
            return torch.tensor(0.0, requires_grad=True, device=loss.device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        for name, param in model.named_parameters():
            if param.grad is not None:
                bad_grad = torch.isnan(param.grad) | torch.isinf(param.grad)
                if bad_grad.any():
                    print(f"⚠️ NaNs/Infs in gradient for {name}. Zeroing them.")
                    param.grad[bad_grad] = 0.0
        return loss

    for epoch in range(20):
        loss = optimizer.step(closure)
        scheduler.step()

        if torch.isfinite(loss):
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
        else:
            print(f"Invalid loss at epoch {epoch}")

        print(f"Epoch {epoch+1:02} | Loss: {loss.item():.5e}")

        if counter >= patience:
            print("Early stopping.")
            break

        # Update the running loss
   #         y_pred = model(t_in,y_in,k)
    #        loss_current = loss_fn(y_pred, y_in,k)
        
        # print loss
    #if epoch % epochs == epochs-1:
     #          print(loss_current.item())
             
     
    # extract the model parameters    
    model_out=list(model.parameters())
    alpha1=(model_out[0]).cpu().detach().numpy()
    omega1=model_out[1].cpu().detach().numpy()
    scale1=model_out[2].cpu().detach().numpy()
    scale2=model_out[3].cpu().detach().numpy()

    # parameter vector
    parms_k=np.concatenate((alpha1,omega1,Softplus(scale1),scale2))
  
    # residuals recordings
    y_resid=y-preds_nodelay(theta=parms_k,t=t,m=k,y=y)
    #y_resid.append(y_resid_temp)
    
    # AIC and BIC recoordings 
    BIC_model=k_map_BIC(resi=y_resid*np.linalg.norm(y), k=k, N=N)
    AIC_model=k_map_AIC(resi=y_resid*np.linalg.norm(y), k=k, N=N)          
    
    parms_k # collect parameters per compontent
    count=k # count number of components in the model
    
    names = ['par', 'residuals','peak_numbers','BIC_model','AIC_model'] # collect parameters, residuals, number of componts, BIC and AIC for output
    out=(parms_k,y_resid[0:k],count,BIC_model,AIC_model)
    named_out = dict(zip(names, out))
    
    return(named_out)