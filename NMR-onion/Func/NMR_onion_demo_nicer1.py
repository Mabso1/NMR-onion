#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:02:59 2022

@author: mathies
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=np.ComplexWarning)# disable plot warnings 
path="/home/mathies/Desktop/nmr onion/NMR onion final"
sys.path.append(path)
from data_import import import_data
from model_caller import onion_model_call,onion_bootstrap_call
from helper_functions import freq_hz,ppm_axis1,t_disrecte,time_series,ppm2hz,hz2ppm,amp_ratio,calc_couplings,result_csv,getoverlaps,count_overlaps
from model_prediction_fuctions import pred,preds_split,single_sinusoid,model_auto_select,par_hat_output
from peak_detection import peak_detection,peaks2hz,signal_points
from onion_filter_experimental import onion_filter,baseline_arPLS,data2fit
#%% import data
data_path="/home/mathies/Desktop/nmr onion/article data experiment 2/DTU800_150922_CHG_MMS052002(1)/DTU800_150922_CHG_MMS052002/10/pdata/1"
sys.path.append(data_path)

# import the data with needed process and aqusition parameters
# import data,data length, sample rate, O1, spectromtor frequency, sweep width (ppm), O1p, zerofill data if needed
data,N,fs,O1,SF,fs_ppm,O1p=import_data(path=data_path,zerofill=False)
#%%
# ROIS
ROI1=np.array([2.4,2.65])
#ROI1=np.array([0.9,1.2])
#ROI2=np.array([3.8,4.1])
#ROI3=np.array([5.2,5.4])
#ROI4=np.array([6.6,7.0])
#ROI5=np.array([7.1,7.5]) # nåede til ROI5 på datasæt 066

# define discrete time (tn) and measure timed (t)
tn=t_disrecte(N)
t=time_series(tn=tn,fs=fs)

# define ppm axis
freq=freq_hz(tn=tn, fs=fs)
ppm_val=ppm_axis1(time=t, O1=O1, fs=fs, fs_ppm=fs_ppm)

# fourier transfrom raw data
fft_raw=np.fft.fftshift(np.fft.fft(data))

target_ROI=ROI1

# plot the data to check it looks properly
#plt.xlabel('ppm')  
#plt.ylabel('Intensity')
#plt.title("NMR Spectrum")  
#plt.plot(ppm_val,fft_raw,color="blue")
#plt.xlim(ROI2[0]-0.2,ROI2[1]+0.2) # zoom x-axis
#plt.ylim(0,0.13*10**7) # zoom y-axis
#plt.show()
#plt.savefig('NMR_Spectrum_ROI1_figure1.pdf')  
#
# plot the time series
#plt.plot(t,data,color="green")
#plt.xlabel('time(s)')  
#plt.ylabel('Intensity')
#plt.title("Full NMR Spectrum FID")  
#plt.savefig('Full_NMR_Spectrum_FID.pdf')  

#filtering the data

high_ppm=target_ROI[1]# set ppm values for region cuts
low_ppm=target_ROI[0]# set ppm values for region cuts

noise_region=[-0.2, -0.1,] # low to high ppm

# apply digital filter to get region of interest and estimated noise level
y_filt,noise_level=onion_filter(low_ppm, high_ppm, noise_region, data, fs, SF, O1p)

# fourier transform filtered data
y_fft_filt=np.fft.fftshift(np.fft.fft((y_filt)))

#_, fft_raw_als, info = baseline_arPLS(fft_raw, lam=1e5, niter=20,
 #                                       full_output=True)
#baseObj=BaselineRemoval(y_fft_filt)
#fft_raw_als=baseObj.ZhangFit()
#
# plot the filtered region of interest
#plt.xlabel('ppm')  
#plt.ylabel('Intensity')
#plt.title("Digital filtered NMR Spectrum ("+str(low_ppm)+"-"+str(high_ppm)+"ppm)",) 
#plt.plot(ppm_val,y_fft_filt)
#plt.plot(ppm_val,y_fft_filt_als,color="red")
#plt.xlim(low_ppm-0.2,high_ppm+0.2)
#plt.ylim(0,0.02*10**7)
#plt.show()
#plt.savefig('digital_filtered_NMR_Spectrum_FID_figure1.pdf')  
#plt.xlim(ppm2hz(high_ppm,SF,O1p)+100, ppm2hz(low_ppm,SF,O1p)-100)


#
# time series filtered region of interest
#plt.xlabel('time(s)')  
#plt.ylabel('Intensity')
#plt.title("Digital filtered NMR Spectrum FID ("+str(low_ppm)+"-"+str(high_ppm)+"ppm)",)  
#plt.plot(t,y_filt)

#ROI4, dataset 1, 1,2
#ROI1, dataset 2, 4,55
#ROI1, dataset 2, 1,2
#ROI5, dataset 1, 1,2

#plt.savefig('digital_filtered_NMR_Spectrum_FID.pdf')  
#plt.xlim(ppm2hz(high_ppm,SF,O1p)+100, ppm2hz(low_ppm,SF,O1p)-100)
#peak detection

#detect peaks in ppm use etiher y=fft_raw (partifuclar for excitation sculpting) if small water signal else use y=y_fft_filt 
omega_ppm=peak_detection(ppm_val=ppm_val, y=y_fft_filt, width=1.0, noise_roi=4, noise_peaks=5, t=t, fs=fs)

# convert detected peaks to hz
omega_hz_filtered=peaks2hz(omega_ppm,high_ppm,low_ppm,SF,O1p) # convert to hz and make sure only peaks within ROI is present
    
# store the detected peak intensitiy values and ppm/hz values (set ppm_out=True for ppm output, else hz output)
ROI_hz_points,ROI_signal_points=signal_points(omega_hz_filtered=omega_hz_filtered, freq=freq, y=y_filt, SF=SF, O1p=O1p,ppm_out=False)

ROI_ppm_points=hz2ppm(np.array(ROI_hz_points),SF,O1p) # point values in ppm


# compute SNR in dB
SNR_db=np.log10(ROI_signal_points/np.std(noise_level))
idx=np.where(np.real(SNR_db)>0)

omega_hz_filtered=omega_hz_filtered[idx]

#%% get correct format for detection
y_fft_filt,y_filt,freq,ppm_val,t,freq,tn=data2fit(ROI=target_ROI, noise_region=noise_region,data_path=data_path)
   
    
# check the peak dection within the region
plt.xlabel('ppm')  
plt.ylabel('Intensity')
plt.title("Peaks detected ("+str(low_ppm)+"-"+str(high_ppm)+"ppm)",) 
plt.plot(ppm_val,y_fft_filt,color="blue")
plt.xlim(low_ppm,high_ppm)
plt.ylim(-100000,1*10**6.0)
for i in range(0,len(omega_hz_filtered)):
    plt.scatter(ROI_ppm_points[idx[0][i]],ROI_signal_points[idx[0][i]],color="red")
#plt.savefig('peaks_detected_ROI1_figure1.pdf')  

#%% model fitting
# lortentzian model
fit1=onion_model_call(model_name="skewed_lorentzian",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))
#%%
# psedou voigt model
fit2=onion_model_call(model_name="skewed_pvoigt",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))
#%%
# genererlized voigt
fit3=onion_model_call(model_name="skewed_genvoigt",omega_hz_filtered=omega_hz_filtered,tn_new=tn,t_new=t,fs_new=fs,y_norm=y_filt/np.linalg.norm(y_filt))
#%% find parameter error of best model
model_compare_BIC=np.array([fit1['BIC_model'],fit2['BIC_model'],fit3['BIC_model']])
model_compare_AIC=np.array([fit1['AIC_model'],fit2['AIC_model'],fit3['AIC_model']])

model_compare_table=pd.DataFrame(({'skewed_lorentzian':(model_compare_BIC[0],model_compare_AIC[0]),
               'skewed_pvoigt':(model_compare_BIC[1],model_compare_AIC[1]),
               'skewed_genvoigt':(model_compare_BIC[2],model_compare_AIC[2]),
    }))

fits=([fit1,fit2,fit3])
#%% inspect model output in hz
par_res,model,k=model_auto_select(model_compare_BIC,model_compare_AIC,fits,omega_hz_filtered,BIC=True)
print(model_compare_BIC)
print(model_compare_AIC)
print(model)
#%%
# get parameter output as vector  
par_hat=par_hat_output(best_model=model,par_res=par_res)

#omega=par_res['omega_est']#-hz_moved # get the decimated results back to original filtered domain
#omega=omega_hz_filtered
#alpha=par_res['alpha_est']
#eta=par_res['eta_est'] # uncomment if model is not lorentzian 
#scale1=par_res['scale'] # uncomment if model is skewed 
#scale2=par_res['scale1']#*0.5 # uncomment if model is skewd

# get FID prediction
y_hat=pred(theta=par_hat,m=k,t=t,y=y_filt/np.linalg.norm(y_filt), model_name=model)

# get individual sinusoids 
single_sinusoids=single_sinusoid(theta=par_hat,m=k,t=t,y=y_filt/np.linalg.norm(y_filt), model_name=model)
A,Z=preds_split(theta=par_hat,m=k,t=t,y=y_filt/np.linalg.norm(y_filt), model_name=model)

# optional parameters amplitude and phase
phases=np.arctan2(np.imag(A),np.real(A)) #phase
amps=np.abs(A) # amplitudes

# define results in frequency domain
y_fft_hat=np.fft.fftshift(np.fft.fft(y_hat)) # fft of FID prediction
y_fft_cut=np.fft.fftshift(np.fft.fft(y_filt/np.linalg.norm(y_filt))) # fft of decimated FID 
y_fft_resi=y_fft_cut-y_fft_hat # residuals of the frequency domain 


# plot the results in hertz
for i in range(0,k):
     plt.plot(freq,(np.fft.fftshift(np.fft.fft(single_sinusoids[i]))),color="red")

plt.plot((freq),y_fft_cut,color="blue")
plt.plot((freq),(y_fft_hat),color="black")
plt.plot(freq,(y_fft_resi),color="orange")
plt.xlim(ppm2hz(high_ppm,SF,O1p), ppm2hz(low_ppm,SF,O1p))
#plt.xlim(-500,-560)
#plt.axvline(-697.47814417-1)
#plt.axvline(-692.40544244)
#plt.axvline(omega_hz_filtered[4],color="green")
#plt.xlim(-2880,-2905)
#plt.xlim(-580,-640)
#plt.xlim(1730,1690)
#plt.xlim(1780,1750)
plt.xlim(2080,2010)
#plt.xlim(-2300,-2360)
#plt.ylim(0,2)
#%% error estimation using wild bootstrap

#_LinAlgError: linalg.solve

# decimate the signal
#y_dem,fs_new,tn_new,t_new,freq_new,hz_moved=signal_decimate(low_ppm=low_ppm,high_ppm=high_ppm, y_filt=y_filt, freq=freq, SF=SF, O1p=O1p, fs=fs)
     
# set eta=eta for non-lorentizain fits 
boot_samples=onion_bootstrap_call(B=1000,par_hat=par_hat,low_ppm=low_ppm,high_ppm=high_ppm,model_name=model, CI_level=0.95,SF=SF,O1p=O1p,freq=freq,fs=fs, t=t, k=k, y=y_filt/np.linalg.norm(y_filt))

#%% collect the output key results

omega=par_res['omega_est'] # frequency estimated in hz

j_mat=calc_couplings(omega=omega, k=k) # coupling pattern matrix 

# plot results in ppm
ppm_val=ppm_axis1(time=t, O1=O1, fs=fs, fs_ppm=fs_ppm)
omega_ppm=hz2ppm(omega_hz=omega, SF=SF, O1p=O1p)
idx_list=getoverlaps(boot_samples,omega_ppm, k)



for i in range(0,k):
    plt.plot(ppm_val,np.fft.fftshift(np.fft.fft(single_sinusoids[i])),color="red")
for i in range(0,len(idx_list)):
    ppm_cord_idx=np.argmax(np.fft.fftshift(np.fft.fft(single_sinusoids[idx_list[i]])))
    plt.scatter(ppm_val[ppm_cord_idx],max(np.fft.fftshift(np.fft.fft(single_sinusoids[idx_list[i]]))),color="green")   
plt.plot(ppm_val,(y_fft_cut),color="blue")
plt.plot(ppm_val,(y_fft_hat),color='black')

#plt.plot(ppm_val,(y_fft_resi),color="orange")
plt.xlim(high_ppm,low_ppm)
plt.xlim(6.93,6.89)
#plt.ylim(0,60)
plt.xlabel('ppm')  
plt.ylabel('Intensity')
#
plt.title("Model results( "+str(low_ppm)+"-"+str(high_ppm)+"ppm)", ) 
#plt.savefig('model_results_ROI1_experiment2_figure1_1_dots.pdf')  
# collect the results peaks (ppm), peaks(hz), amplitude ratios (freq domain), coupling pattern matrix 
#%%


#%% extract the key results

result_csv(single_sinusoids,boot_samples,k,omega,idx_list,SF,j_mat,O1,O1p)
#xx=boot_samples['omega_boot']
#%%

# couptling pattern matrix 
j_mat=calc_couplings(omega=omega, k=k)
 

# amplitudes ratio
amp_ratios=amp_ratio(peaks=single_sinusoids, k=k)

boot_samples['omega_std'].index=range(k)

#table of uncertainties
res_uncertantinty=pd.DataFrame(({'omega_hz_error':boot_samples['omega_std'],
                                'omega_hz_CI_lower':boot_samples['omega_CI_lower']+O1,
                                'omega_hz_CI_upper':boot_samples['omega_CI_upper']+O1,
                                'omega_ppm_error':hz2ppm(omega_hz=boot_samples['omega_boot'], SF=SF, O1p=O1p).T.std(),
                                'omega_ppm_CI_lower':hz2ppm(omega_hz=boot_samples['omega_CI_lower'], SF=SF, O1p=O1p),
                                'omega_ppm_CI_upper':hz2ppm(omega_hz=boot_samples['omega_CI_upper'], SF=SF, O1p=O1p)
    }))
res_uncertantinty.index = range(k)


# table of regional results
res_region=pd.DataFrame(({'amp_ratio':amp_ratios,
                         'omega_hz':omega+O1,
                         'omega_ppm':omega_ppm,
                         'overlaps':xx,
                         }))

res_combo=pd.concat([res_region,res_uncertantinty,j_mat],axis=1)
    
#%% manually plot the histrogram overlaps for overlapping CIs
omega_samples=boot_samples['omega_boot']
omega_samples.index = range(k)

df_max=omega_samples.T.max(axis=1).max()+O1
df_min=omega_samples.T.min(axis=1).min()+O1

sub_roi_CIl=res_combo['omega_hz_CI_lower']
sub_roi_CIu=res_combo['omega_hz_CI_upper']

sub_roi_CIl.index = range(k)
sub_roi_CIu.index = range(k)

_, bins, _ = plt.hist(omega_samples.T[idx_list[0]], bins=500,range=(df_min,df_max))
plt.title("boostrap samples of two peaks")
plt.xlabel("Hz")
plt.ylabel("Emperical density")
plt.xlim(-2340+O1,-2350+O1)  
_= plt.hist(omega_samples.T[idx_list[14]]+O1,color="r", bins=bins,density=True)  
_= plt.hist(omega_samples.T[idx_list[15]]+O1,ec="b",fc="none", bins=bins,density=True)    
plt.axvline(sub_roi_CIl[idx_list[15]],color="green")
plt.axvline(sub_roi_CIu[idx_list[15]],color="green")
plt.axvline(sub_roi_CIl[idx_list[14]],color="black")
plt.axvline(sub_roi_CIu[idx_list[14]],color="black")
plt.savefig('peaks_overlaps_subROI1_figure1.pdf')  