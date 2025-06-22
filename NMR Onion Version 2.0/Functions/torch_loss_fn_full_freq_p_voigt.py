#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:47:14 2022

@author: mathies
"""
import torch
import numpy as np
import warnings
from scipy.special import expit as expit_sci
from torch.special import expit
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
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
# m= number of components in y_t=sum^{k}_{i}=1 (A_i*exp(2j*pi*omega_i*t+phi_i)*exp(-alpha_i*t**eta_i)
# y= time series data (FID)
### outputs ##
# return predicted model in the time domian (FID)
def preds_nodelay(theta,t,m,y):
     # signal pole matrix
     Z_poles = np.exp(np.outer(t, (1j * 2 * np.pi * theta[1 * m:2 * m])))
     # weigthed decay
     decay=(1-expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer(t,-theta[3*m:3*m+1]*np.exp(1j*expit_sci(theta[3*m+1:4*m+1])*np.pi/4)*expit_sci(theta[0 * m:1 * m])))+(expit_sci(theta[2 * m:3 * m]))*np.exp(np.outer((t)**2,-theta[3*m:3*m+1]*np.exp(1j*theta[3*m+1:4*m+1])*expit_sci(theta[0 * m:1 * m])))
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
def onion_pvoigt(omega, tn, t, fs, y):
    import torch
    import numpy as np
    from torch.nn.functional import softplus
    from torch.special import expit
    from torch.autograd import Variable
    
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

    class SinusoidModel(torch.nn.Module):
        def __init__(self, p, k):
            super().__init__()
            self.a = torch.nn.Parameter(p[0:k].clone().to(torch.float64))
            self.f = torch.nn.Parameter(p[k:2 * k].clone().to(torch.float64))
            self.e = torch.nn.Parameter(p[2 * k:3 * k].clone().to(torch.float64))
            self.scale = torch.nn.Parameter(p[3 * k:3 * k + 1].clone().to(torch.float64))
            self.scale2 = torch.nn.Parameter(p[3 * k + 1:4 * k + 1].clone().to(torch.float64))

        def forward(self, t,y,k):
            # signal pole matrix
             Z_poles= torch.exp(torch.outer(t_in,1j * 2 * torch.pi * self.f))#-SP(self.a)))
             # weigthed decay
             decay=(1-expit(self.e))*torch.exp(torch.outer(t_in,-SP(self.scale)*expit(self.a)*torch.exp(1j*self.scale2)))+expit(self.e)*torch.exp(torch.outer(t_in**2,-SP(self.scale)*torch.exp(1j*self.scale2)*expit(self.a)))
             
            # model matrix Z(theta)
             Z=Z_poles*decay
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
        
             return pred,A

    

    model = SinusoidModel(p, k).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.9, max_iter=200,
                                  tolerance_grad=1e-5, tolerance_change=1e-9,
                                  history_size=100, line_search_fn='strong_wolfe')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.70)

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

    with torch.no_grad():
        out = list(model.parameters())
        alpha1 = out[0].cpu().numpy()
        omega1 = out[1].cpu().numpy()
        eta1 = out[2].cpu().numpy()
        scale1 = out[3].cpu().numpy()
        scale2 = out[4].cpu().numpy()

        parms_k = np.concatenate((alpha1, omega1, eta1, np.log1p(np.exp(scale1)), scale2))


        y_resid_temp = y - preds_nodelay(parms_k, t, k, y)
        y_norm = np.linalg.norm(y)

        def AIC(resi, k, N):
            SSE = np.real(np.dot(np.conj(resi), resi))
            MSE = SSE / N
            loglik = -N / 2 * ((1 + np.log(2 * np.pi)) + np.log(MSE / N))
            return -2 * loglik + (1 + 2 * 4 * k)

        def BIC(resi, k, N):
            SSE = np.real(np.dot(np.conj(resi), resi))
            MSE = SSE / N
            loglik = -N / 2 * ((1 + np.log(2 * np.pi)) + np.log(MSE / N))
            return -2 * loglik + (1 + 2 * 4 * k * np.log(N))

        return {
            'par': parms_k,
            'residuals': y_resid_temp,
            'peak_numbers': k,
            'BIC_model': BIC(y_resid_temp * y_norm, k, N),
            'AIC_model': AIC(y_resid_temp * y_norm, k, N)
        }