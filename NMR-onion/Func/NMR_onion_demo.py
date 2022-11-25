#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:02:59 2022

@author: mathies
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings 
path="/home/mathies/Desktop/nmr onion/NMR onion final"
sys.path.append(path)
from data_import import *
from helper_functions import *
from model_prediction_fuctions import *
from peak_detection import *

warnings.filterwarnings("ignore", category=np.ComplexWarning)# disable plot warnings 

#%% import data
data_path="/home/mathies/Desktop/nmr onion/real data benchmarking/DTU800_150922_CHG_MMS052002(1)/DTU800_150922_CHG_MMS052002/10/pdata/1"
sys.path.append(data_path)

# import the data with needed process and aqusition parameters
# import data,data length, sample rate, O1, spectromtor frequency, sweep width (ppm), O1p
data,N,fs,O1,SF,fs_ppm,O1p=import_data(path=data_path,zerofill=True)

# define discrete time (tn) and measure timed (t)
tn=t_disrecte(N)
t=time_series(tn=tn,fs=fs)

# define ppm axis
ppm_val=ppm_axis1(time=t, O1=O1, fs=fs, fs_ppm=fs_ppm)

# plot the data to check it looks properly
plt.plot(ppm_val,np.fft.fftshift(np.fft.fft(data)),color="blue")
plt.show()

# plot the time series
plt.plot(t,data,color="blue")

#%% filtering the data
from nmrespy import ExpInfo
from nmrespy.freqfilter import Filter

high_ppm=2.65 # set ppm values for region cuts
low_ppm=2.4 # set ppm values for region cuts

expinfo = ExpInfo(
    dim=1,
    sw=fs,
    sfo=SF,
    nuclei="1H",
    default_pts=N,
)

fid = data
region = [ppm2hz(high_ppm,SF,O1p), ppm2hz(low_ppm,SF,O1p),]
noise_region = [ppm2hz(-0.2,SF,O1p), ppm2hz(-0.1,SF,O1p),]
filterobj = Filter(
    fid,
    expinfo,
    region,
    noise_region,
)

cut_filtered_fid1, cut_expinfo1 = filterobj.get_filtered_fid(cut_ratio=None)
cut_tp1, = cut_expinfo1.get_timepoints(pts=cut_filtered_fid1.shape)
moved=cut_expinfo1._offset[0]

fs_new=cut_expinfo1._sw[0]
tn_new=np.arange(0.,len(cut_filtered_fid1),1.)
t_new=cut_expinfo1.get_timepoints()[0]
t_new=t_new

freq_new=freq_hz(tn_new, fs_new)
ppm_val_new=ppm_omega(omega_hz=freq_new, SF=SF, O1p=O1p)

y_filt=cut_filtered_fid1

plt.plot(freq_new,np.fft.fftshift(np.fft.fft((y_filt))))
plt.xlim(ppm2hz(high_ppm,SF,O1p), ppm2hz(low_ppm,SF,O1p))
#%% peak detection
omega_ppm=peak_detection(ppm_val=ppm_val_new, y=np.fft.fftshift(np.fft.fft(y_filt)), width=1.0, noise_roi=4, noise_peaks=2, t=t_new, fs=fs)
omega_hz=ppm2hz(ppm_value=omega_ppm, SF=SF, O1p=O1p) # conver to hz

#%% check the peak dection within the region
plt.plot((freq_new),np.fft.fftshift(np.fft.fft(y_filt)),color="orange")
plt.xlim(ppm2hz(high_ppm,SF,O1p), ppm2hz(low_ppm,SF,O1p))
#plt.ylim(0,10**6.7)

for i in range(0,len(omega_ppm)):
    plt.axvline(-omega_hz[i])

# make sure that peaks are only within the region and save as omega_hz_filtered
idx=np.where( (-omega_hz>ppm2hz(low_ppm,SF,O1p)) &  (-omega_hz<ppm2hz(high_ppm,SF,O1p)) )
omega_hz_filtered=-omega_hz[idx]
#%% model fitting
path="/home/mathies/Desktop/nmr onion"
sys.path.append(path)
from torch_loss_fn_full_freq_lor import *

start =time.time()
fit1=flex_decay_fitting_onion_nodelay_refine_lor(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_filt/np.linalg.norm(y_filt))
print((time.time() - start))
#%% 
from torch_loss_fn_full_freq_p_voigt import *

start =time.time()
fit2=flex_decay_fitting_onion_nodelay_refine_pvoigt(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_filt/np.linalg.norm(y_filt))
print((time.time() - start))
#%%
path="/home/mathies/Desktop/nmr onion"
sys.path.append(path)
from torch_loss_fn_full_freq_gen_voigt import *

start =time.time()
fit3=flex_decay_fitting_onion_nodelay_refine_genvoigt(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_filt/np.linalg.norm(y_filt))
print((time.time() - start))
#%% find parameter error of best model
model_compare=np.array([fit1['BIC_model'],fit2['BIC_model'],fit3['BIC_model']])
print(model_compare)
#%% inspect model output
from model_prediction_fuctions import *
model="skewed_genvoigt"
k=len(omega_hz_filtered)
par_res=par_est(fit=fit3, k=k, model_name=model)

omega=par_res['omega_est']
alpha=par_res['alpha_est']
eta=par_res['eta_est']  # uncomment if model is not lorentzian 
scale1=par_res['scale'] # uncomment if model is skewed 
scale2=par_res['scale1'] # uncomment if model is skewd

# set par_hat in the following order: alpha, omega,eta,scale1 and scale 2 
par_hat=np.concatenate(np.array([alpha,omega,eta,scale1,scale2])) # collect the parameters

# get FID prediction
y_hat=pred(theta=par_hat,m=k,t=t_new,y=y_filt/np.linalg.norm(y_filt), model_name=model)

# get individual sinusoids 
single_sinusoid=single_sinusoid(theta=par_hat,m=k,t=t_new,y=y_filt/np.linalg.norm(y_filt), model_name=model)
A,Z=preds_split(theta=par_hat,m=k,t=t_new,y=y_filt/np.linalg.norm(y_filt), model_name=model)

# optional parameters amplitude and phase
phases=np.arctan2(np.imag(A),np.real(A)) #phase
amps=np.abs(A) # amplitudes

# define results in frequency domain
y_fft_hat=np.fft.fftshift(np.fft.fft(y_hat)) # fft of FID prediction
y_fft_cut=np.fft.fftshift(np.fft.fft(y_filt/np.linalg.norm(y_filt))) # fft of decimated FID 
y_fft_resi=y_fft_cut-y_fft_hat # residuals of the frequency domain 

# plot the results in hertz
for i in range(0,k):
     plt.plot(freq_new,(np.fft.fftshift(np.fft.fft(single_sinusoid[i]))),color="red")
plt.plot((freq_new),y_fft_cut,color="blue")
plt.plot((freq_new),(y_fft_hat),color="black")
plt.plot(freq_new,(y_fft_resi),color="orange")
#plt.xlim(ppm2hz(high_ppm,SF,O1p), ppm2hz(low_ppm,SF,O1p))
plt.xlim(-2425, -2480)
plt.ylim(0,20)
#%%
test=parameter_error(y_hat=y_hat, par_hat=par_hat, fs=fs_new, t=t_new, k=k, y_obs=y_filt/np.linalg.norm(y_filt), CI_level=0.95)

#%% collect the output key results

# plot results in ppm
ppm_val=ppm_axis1(time=t_new, O1=O1, fs=fs, fs_ppm=fs_ppm)

for i in range(0,k):
    plt.plot(ppm_val,(np.fft.fftshift(np.fft.fft(single_sinusoid[i]))),color="red")
plt.plot(ppm_val,(y_fft_cut),color="blue")
plt.plot(ppm_val,(y_fft_hat),color='black')
plt.plot(ppm_val,(y_fft_resi),color="orange")
plt.xlim(high_ppm,low_ppm)
#plt.ylim(-10**4,10**6.5)

# collect the results peaks (ppm), peaks(hz), amplitude ratios (freq domain), coupling pattern matrix 



