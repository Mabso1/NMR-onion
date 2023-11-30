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
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
#import pytorch_warmup as warmup


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
# m= number of components in y_t=sum^{k}_{i}=1 (A_i*exp(2j*pi*omega_i*t+phi_i)*exp(-alpha_i*t**eta_i)
# y= time series data (FID)
### outputs ##
# return predicted model in the time domian (FID)
def preds_nodelay(theta,t,m,y):
     # signal pole matrix
     Z_poles = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m])))
     # weigthed decay
     decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))
   # model matrix Z(theta)
     Z=Z_poles*decay
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
# pred= vector of predicted parameters (alpha,omega,eta)=(decay,frequency,weigthing constant)
# y_in= time series data (FID) stored as a tensor
### outputs ###
# returns loss value as resid@resid**H, H=conjugated and resid=residuals
def loss_fn(pred,y_in,k):
    resid= torch.reshape(y_in,(len(y_in),1)) - pred[0] # residuals
    resid=resid.flatten()
    
    obj=torch.real(torch.dot(torch.conj(resid),resid)) # real dot product of conjugated residuals and residuals
    
    phases = torch.arctan2(torch.imag(pred[1]),torch.real(pred[1]))
    mu = torch.sum(phases.flatten()) / k
    obj += torch.sum((phases.flatten() - mu) ** 2) / (torch.pi*k)
    
    
    
    return obj

### k_map_AIC ###
# defines the Akaike information criterion (AIC) for model selection/selection of number of components
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
    K_map_temp=-2*(loglik_temp)+(1+2*4*k) # AIC formula= -2*log_lik/N+2*p/N, p=number of parameters (3 ekstra added per compontent)
   
    return(K_map_temp)


### k_map_BIC ###
# defines the Bayesian information criterion (BIC) for model selection/selection of number of components
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
    K_map_temp=-2*(loglik_temp)+(1+2*4*k*np.log(N)) # BIC formula= -2*log_lik/N+(2*p*log(N))/N, p=number of parameters (3 ekstra added per compontent)
   
    return(K_map_temp)

#obj_grad=grad(obj_nodelay)
#obj_hess=hessian(obj_nodelay)
### flex_decay_fitting_onion_nodelay ###
# defines the sequential fitting algorithmn of NMR-onion used to deconvolute an NMR FID
### inputs ###
# omega= a list of detected peaks found by the detection algorithmn 
# tn= discrete time series
# t= time series
# y= digitally filtered data filtered data (preferably normalized) 
### outputs ###
# returns a list containing peak numbers, parameters, AIC, BIC and current SNR
def onion_pvoigt(omega,tn,t,fs,y):
    N=len(y)
    y_resid=[] # residual list
    parm=[]
    SP = torch.nn.Softplus()

    
    init_f=omega
    k=len(init_f)    
    init_a=np.repeat(0.0,k)
    init_e=np.repeat(0.0,k)
    init_scale=0.0
    init_scale2=np.repeat(0.0,k)
            
    init_a=np.hstack(init_a)
    init_f=np.hstack(init_f)
    init_e=np.hstack(init_e)

    
    init2=np.hstack((init_a,init_f,init_e,init_scale,init_scale2))
    #init2=np.asarray((init_a,init_f,init_e,init_scale,init_scale2))
   # init2=np.hstack(init2)

        
    # convert time (t) and FID values (y) to tensors       
    t_in=torch.from_numpy(t)
    y_in=torch.from_numpy(y)
    
    
    class MyDataset(Dataset):
        def __init__(self, x, y):
            super(MyDataset, self).__init__()
            assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
            self.x = x
            self.y = y
        
        
        def __len__(self):
            return self.y.shape[0]
        
        def __getitem__(self, index):
            return self.x[index], self.y[index]
    
    
    train_data=MyDataset(t_in,y_in)
    train_load=DataLoader(train_data,batch_size=N,shuffle=False,num_workers=4)
      
    # convert inital value array to tensor and attach gradient
    p=Variable(torch.from_numpy(init2),requires_grad=True).type(torch.FloatTensor) # convert to tensor
        
    class sinusoid_refit(torch.nn.Module):
            def __init__(self):
                
                # define number of paameters a =decay, f=frequency and e=flexibility constant
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(p[0:k],dtype=torch.float64))
                self.f = torch.nn.Parameter(torch.tensor(p[k:2*k],dtype=torch.float64))
                self.e = torch.nn.Parameter(torch.tensor(p[2*k:3*k],dtype=torch.float64))
                self.scale=torch.nn.Parameter(torch.tensor(p[3*k:3*k+1],dtype=torch.float64))
                self.scale2=torch.nn.Parameter(torch.tensor(p[3*k+1:4*k+1],dtype=torch.float64))

                
            def forward(self, t_in,y_in,k):
                # signal pole matrix
                 Z_poles= torch.exp(torch.outer(t_in,1j * 2 * torch.pi * self.f))#-SP(self.a)))
                 # weigthed decay
                 decay=(1-expit(self.e))*torch.exp(torch.outer(t_in,-SP(self.scale)*expit(self.a)*torch.exp(1j*self.scale2)))+expit(self.e)*torch.exp(torch.outer(t_in**2,-SP(self.scale)*torch.exp(1j*self.scale2)*expit(self.a)))
                 
                # model matrix Z(theta)
                 Z=Z_poles*decay
                 # small diagnonal to aviod instability
                 diag_small=torch.eye(k*1)*(1j+1)*10**-6
                 # complex amplitudes
               #  L = torch.linalg.cholesky(torch.conj(Z).T@Z+diag_small)

                # A=torch.cholesky_solve(torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1)), L,upper=False)
                 A=torch.linalg.solve(torch.conj(Z).T@Z+diag_small,torch.conj(Z).T@torch.reshape(y_in,(len(y_in),1))) 
                 # prediction of the model
                 pred=Z@A
            
                 return pred,A
        
    model = sinusoid_refit()
        
    # call the optimizer            
    optimizer=torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.70)
  #  warmup_scheduler = warmup.LinearWarmup(optimizer,1)
    # make the forward pass via a loop
    epochs=20 # full batch so only one epoch needed 
    for epoch in range(epochs):
        running_loss=0.0
        for steps, (t_batch,y_batch) in enumerate(train_load): 
            
             def closure():
               if torch.is_grad_enabled():
                   optimizer.zero_grad()
               y_pred = model(t_batch,y_batch,k)
               loss = loss_fn(y_pred, y_batch,k)           
               if loss.requires_grad:
                   loss.backward()
                   print('loss:', loss.item())
               return loss
        # Update weights
           #  torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0,norm_type=2)
            # with warmup_scheduler.dampening():
             
             optimizer.step(closure)
        loss=closure()
        running_loss+=loss.item()
        scheduler.step()
        print(f"Epoch:{epoch+1:02}/20 Loss: {running_loss:.5e}")
             #loss_total+=loss_out.item()
             #print('total_loss-->',loss_total,epoch)
        # Update the running loss
       # y_pred = model(t_in,y_in,k)
        #loss_current = loss_fn(y_pred, y_in,k)
        
        # print loss
    #if epoch % epochs == epochs-1:
     #         print(loss_current.item())
        
    model_out=list(model.parameters())
    alpha1=(model_out[0]).detach().numpy()
    omega1=model_out[1].detach().numpy()
    eta1=(model_out[2]).detach().numpy()
    scale1=(model_out[3]).detach().numpy()
    scale2=(model_out[4]).detach().numpy()

    # parameter vector
    parms_k=np.concatenate((alpha1,omega1,eta1,Softplus(scale1),scale2))

    # residuals recordings
    y_resid_temp=y-preds_nodelay(theta=parms_k,t=t,m=k,y=y)
    y_resid.append(y_resid_temp)
    
    # BIC and AIC reccording
    BIC_model=k_map_BIC(resi=y_resid_temp*np.linalg.norm(y), k=k, N=N)
    AIC_model=k_map_AIC(resi=y_resid_temp*np.linalg.norm(y), k=k, N=N)          
    
    parm.append(parms_k) # collect parameters per compontent
    count=k # count number of components in the model
    names = ['par', 'residuals','peak_numbers','BIC_model','AIC_model'] # collect parameters and residuals for output
    out=(parm[0:k],y_resid[0:k],count,BIC_model,AIC_model)
    named_out = dict(zip(names, out))
    
    return(named_out)
