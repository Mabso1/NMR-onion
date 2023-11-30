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
    Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*theta[2 * m+1:2 * m+2])*Softplus(theta[0:m]))))
     
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
     Z = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m]-theta[2 * m:2 * m+1]*np.exp(1j*theta[2 * m+1:3 * m+1])*expit_sci(theta[0:m]))))
     
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
    N=len(y) # length of data set
    y_resid=[] # residual list
    parm=[] # parameters list
    SP = torch.nn.Softplus() # softplus function
    
    init_f=omega # inital frequency values
    k=len(init_f) # number of components 
    #init_a=grid_search_alpha(omega, y, t, k) # grid search alpha value
    #print("initial alpha found")
    
    init_a=np.repeat(0.0,k) # inital decay rate - set this to a high number
    init_scale=0.0 # inital scale as 1.0=no scaling (controls width scale of all peaks)
    init_scale2=np.repeat(0.0,k) # inital exp(scale) as 0.0=no scaling (controls skewness of peaks)    
        
 #   init_a=np.hstack(init_a) # collect inital decay rates
  #  init_f=np.hstack(init_f) # collect inital frequency values
 
    
    
    init2=np.hstack((init_a,init_f,init_scale,init_scale2)) # set all parameters in an array
   # init2=np.hstack(init2) # stack the array to correct dims
     
        
    # convert time (t) and FID values (y) to tensors       
    t_in=torch.from_numpy(t)
    y_in=torch.from_numpy(y)
    train_data=TensorDataset(t_in,y_in)
    train_load=DataLoader(train_data,batch_size=N,shuffle=False,num_workers=16,multiprocessing_context=get_context('fork'))
    
    # convert inital value array to tensor and attach gradient
    p=Variable(torch.from_numpy(init2),requires_grad=True).type(torch.FloatTensor) # convert to tensor
    
    # generate the pytorch model class
    class sinusoid_refit(torch.nn.Module):
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
                 diag_small=(torch.eye(k*1)*(1j+1)*10**-6).cuda()
                 # complex amplitudes
               #  L = torch.linalg.cholesky(torch.conj(Z).T@Z+diag_small)

                # A=torch.cholesky_solve(torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1)), L,upper=False)
                 
                 A=torch.linalg.solve(torch.conj(Z).T@Z+diag_small,torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1))) 
                 # prediction of the model
                 pred=Z@A
            
                 return pred,A # return predicions and complex amplitudes 
        
    model = sinusoid_refit().cuda() # define model
    
    #y_pred = model(t_in,y_in,k)
    #loss_current = loss_fn(y_pred, y_in,k)
   # print(loss_current)
    
    # call the LBFGS optimizer from pytorch         
    optimizer=torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    
   # Define the early stopping criteria
    patience = 2  # Stop training if the loss doesn't improve for 2 epochs
    best_loss = float('inf')
    counter = 0
    
    def closure():
    #  if torch.is_grad_enabled():
      optimizer.zero_grad()
      y_pred = model(t_batch,y_batch,k)
      loss = loss_fn(y_pred, y_batch,k)           
     # if loss.requires_grad:
      loss.backward()
      print('loss:', loss.item())
      return loss
  
    
    # make the forward pass via a loop
    epochs=20 # full batch so only one epoch needed 
    for epoch in range(epochs):
        running_loss=0.0
        for steps, (t_batch,y_batch) in enumerate(train_load): 
             t_batch,y_batch=t_batch.cuda(),y_batch.cuda()
             
             optimizer.step(closure)
        loss=closure()
        running_loss+=loss.item()
        scheduler.step()
        
        # Check if the training loss has improved
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        
    # Stop training if the loss hasn't improved for `patience` epochs
        if counter >= patience:
            print(f'Early stopping after epoch {epoch}')
            break
        
        print(f"Epoch:{epoch+1:02}/20 Loss: {running_loss:.5e}")

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
    y_resid_temp=y-preds_nodelay(theta=parms_k,t=t,m=k,y=y)
    y_resid.append(y_resid_temp)
    
    # AIC and BIC recoordings 
    BIC_model=k_map_BIC(resi=y_resid_temp*np.linalg.norm(y), k=k, N=N)
    AIC_model=k_map_AIC(resi=y_resid_temp*np.linalg.norm(y), k=k, N=N)          
    
    parm.append(parms_k) # collect parameters per compontent
    count=k # count number of components in the model
    
    names = ['par', 'residuals','peak_numbers','BIC_model','AIC_model'] # collect parameters, residuals, number of componts, BIC and AIC for output
    out=(parm[0:k],y_resid[0:k],count,BIC_model,AIC_model)
    named_out = dict(zip(names, out))
    
    return(named_out)