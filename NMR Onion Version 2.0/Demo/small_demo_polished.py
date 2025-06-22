#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:00:16 2025

@author: mabso
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=np.ComplexWarning)# disable plot warnings 
path="/home/mabso/Desktop/NMR_Onion_upgraded/NMR Onion Version 2.0/Functions" # set to the path in which NMR onion folder is stored
sys.path.append(path)
#import nmrglue as ng
from plotting_functions import plot_results, plot_model_bootstrap
from data_import import import_data
from model_caller import onion_model_call,onion_bootstrap_call
from helper_functions import result_csv_noboot, freq_hz,ppm_axis1,t_disrecte,time_series,result_csv
from model_prediction_fuctions import model_auto_select,spectrum_prediction
from peak_detection import onion_peak_detection,integrate_peaks
from onion_filter_experimental import onion_filter_expaned

#%%
# set your own data_path
data_path="/home/mabso/Desktop/NMR_Onion_upgraded/NMR Onion Version 2.0/Data/DTU800_150922_CHG_MMS052002/10/pdata/1"
sys.path.append(data_path)

# import the data with needed process and aqusition parameters
# import data,data length, sample rate, O1, spectromtor frequency, sweep width (ppm), O1p, zerofill data if needed
data,N,fs,O1,SF,fs_ppm,O1p=import_data(path=data_path,zerofill=True)

# define discrete time (tn) and measure timed (t)
tn=t_disrecte(N)
t=time_series(tn=tn,fs=fs)

# define ppm axis
freq=freq_hz(tn=tn, fs=fs)
ppm_val=ppm_axis1(time=t, O1=O1, fs=fs, fs_ppm=fs_ppm)

# fourier transfrom raw data

y_fft_raw=np.fft.fftshift(np.fft.fft(data))


# check the data import works by plotting
plt.xlabel('ppm')  
plt.ylabel('Intensity')
plt.title("NMR Spectrum")  
plt.plot(ppm_val,y_fft_raw,color="blue")
plt.xlim(1.0,8.0) # zoom x-axis
plt.ylim(-100000,0.13*10**7) # zoom y-axis
plt.show()

# plot the time series
plt.title("FID")  
plt.plot(t,data,color="green")
plt.xlabel('time(s)')  
plt.ylabel('Intensity')
plt.show()

#%%
ROI1=np.array([3.35,3.6])
target_ROI=ROI1

high_ppm=target_ROI[1]# set ppm values for region cuts
low_ppm=target_ROI[0]# set ppm values for region cuts

noise_region=[-0.2, -0.1,] # low to high ppm


# apply digital filter to get region of interest and estimated noise level
y_filt,noise_level,baseline_estimate=onion_filter_expaned(low_ppm, high_ppm, noise_region, data, fs, SF, O1p,offset=True,baseline_raw=False,minimum_filter=False)

y_fft_filt=np.fft.fftshift(np.fft.fft((y_filt)))

plt.xlabel('ppm')  
plt.ylabel('Intensity')
plt.title("Filtered NMR Spectrum")  
plt.plot(ppm_val,y_fft_filt,color="green")


plt.xlim(low_ppm,high_ppm) # zoom x-axis
plt.ylim(-49000,0.09*10**8) # zoom y-axis
plt.show()
#%%

high_ppm=target_ROI[1]# set ppm values for region cuts
low_ppm=target_ROI[0]# set ppm values for region cuts

ylim1=0
ylim2=0.7*10**7.0


omega_hz_filtered=onion_peak_detection(width=0.8, noise_peaks=15, ylim1=ylim1, ylim2=ylim2, y_fft_filt=y_fft_filt,
                                       low_ppm=low_ppm, high_ppm=high_ppm, ppm_val=ppm_val,
                                       t=t, fs=fs,freq=freq,y_filt=y_filt,noise_level=noise_level,SF=SF,O1p=O1p)

#%%
fit1=onion_model_call(model_name="skewed_lorentzian",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))

#%%
fit2=onion_model_call(model_name="skewed_pvoigt",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))

#%%
fit3=onion_model_call(model_name="skewed_genvoigt",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))


#%%
model_compare_BIC=np.array([fit1['BIC_model'],fit2['BIC_model'],fit3['BIC_model']])
model_compare_AIC=np.array([fit1['AIC_model'],fit2['AIC_model'],fit3['AIC_model']])

model_compare_table=pd.DataFrame(({'skewed_lorentzian':(model_compare_BIC[0],model_compare_AIC[0]),
               'skewed_psedou_voigt':(model_compare_BIC[1],model_compare_AIC[1]),
               'skewed_generlized_voigt':(model_compare_BIC[2],model_compare_AIC[2]),
    }))

# make a list of fits
fits=([fit1,fit2,fit3])

par_res,model_name,k=model_auto_select(model_compare_BIC,model_compare_AIC,fits,omega_hz_filtered,BIC=True)
print(model_compare_BIC) # print BIC values, 0 inidicates no covergence
print(model_compare_AIC) # print AIC values, 0 indicates no convergence
print(model_name) # print the best model name

#%% plot results (good idea to set your save directory to the same everywhere!)
data_path="/home/mabso/testing" # set your current wd here!
sys.path.append(data_path)

par_hat,y_hat,single_sinusoids,amps,phases=spectrum_prediction(model_name,par_res,y_filt,t,k,optional_par=False)

ylim1=0
ylim2=20
xlim1=1100
xlim2=2250

xlim1_ppm=ROI1[0]
xlim2_ppm=ROI1[1]

# plot the results and save to your chosen dir with save_dir choose either pdf or png format
plot_results(xlim1_ppm,xlim2_ppm, ylim1,ylim2,xlim1,xlim2,y_hat=y_hat,y_filt=y_filt,
    t=t,freq=freq,ppm_val=ppm_val,k=k,single_sinusoids=single_sinusoids,time_domain=True,freq_domain=False,
    freq_domain_ppm=True,show_peaks=True, residuals=True,save_dir=data_path,save_format="pdf")

#%%
#extract integrals per deconvoulted peak
omega_ppm, omega_hz, int_list= integrate_peaks(
    xlim1_ppm=xlim1_ppm, xlim2_ppm=xlim2_ppm,
    ylim1=ylim1, ylim2=ylim2,
    single_sinusoids=single_sinusoids,
    ppm_val=ppm_val,
    par_res=par_res,
    SF=SF,
    O1p=O1p,
    plot=True,
    save=True,
    save_dir=data_path,
    filename='auto_integration',
    format='pdf'  # or 'png'
)
#%%
# extract and save your output (currently sent to the fold specified by the plot functions)
data_noboot=result_csv_noboot(single_sinusoids,k,int_list,omega_hz,SF,O1,O1p,save_output=True) # save set to current wd

#%% run the bootstrap function this takes a while!
boot_samples=onion_bootstrap_call(parallel=False,cores=1,B=50,par_hat=par_hat,low_ppm=low_ppm,high_ppm=high_ppm,model_name=model_name, CI_level=0.95,SF=SF,O1p=O1p,freq=freq,fs=fs, t=t, k=k, y=y_filt/np.linalg.norm(y_filt))

#%% save your bootstrap results set a new working directory before doing this!
new_path="/home/mabso/testing/bootstrap" # suggestion is to add an addtional folder for your current wd called bootstrap
sys.path.append(new_path)

data_boot=result_csv(single_sinusoids,boot_samples,k,int_list,omega_hz,SF,O1,O1p,save_output=True) # save set to current wd

#%% plot your bootstrap results
plot_model_bootstrap(
    ppm_val=ppm_val,
    single_sinusoids=single_sinusoids,
    y_hat=y_hat,
    y_filt=y_filt,
    xlim1_ppm=xlim1_ppm,
    xlim2_ppm=xlim2_ppm,
    ylim1=ylim1,
    ylim2=ylim2,
    k=k,
    boot_samples=boot_samples,
    omega_ppm=omega_ppm,
    save_dir=new_path,
    save_format="pdf"  # or 'png'
)


