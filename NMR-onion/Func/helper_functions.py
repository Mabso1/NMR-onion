#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:26:41 2022

@author: mathies
"""
import numpy as np
import pandas as pd
import math


# collect of various smaller functions

# define discrete time points
def t_disrecte(N):
    tn=np.arange(0, N,1)
    return tn

# define time series
def time_series(tn,fs):
    t=1/fs*tn
    return t

# define frequency axis
def freq_hz (tn,fs):
    m=len(tn)
    fx=(tn-math.floor(m/2))*fs/(m-1)
    return(fx)

# make ppm axis 
def ppm_axis1(time,O1,fs,fs_ppm):
    N=len(time)
    interval_ppm=fs_ppm/(N-1)
    O1idx=round( (N+1)/2+O1*(N-1)/fs )
    start=O1idx-1
    end=O1idx-N
    ppm=np.linspace(start,end,N)*interval_ppm
    ppm_flip=np.flip(ppm)
    return ppm_flip

# convert the frequency parameters to bruker ppm format 
def hz2ppm(omega_hz,SF,O1p):
    omega_ppm=omega_hz/SF+O1p
    return(omega_ppm)

# convert from ppm bruker format to standard hz
def ppm2hz(ppm_value,SF,O1p):
    omega_hz=(ppm_value-O1p)*SF
    return omega_hz

# compute the height ratio between peaks
def amp_ratio (peaks,k):
    ratio_collect=[]
    max_peak_height=np.real(np.max( (np.fft.fftshift(np.fft.fft(peaks)))))
    
    for i in range(0,k):
        max_height_idx=np.argmax(np.array([ abs(np.max(np.real( (np.fft.fftshift(np.fft.fft(peaks[i])))))),abs(np.min(np.real((np.fft.fftshift(np.fft.fft(peaks[i]))))))]))
        if (max_height_idx==1):
            max_height=np.min(np.real(np.fft.fftshift(np.fft.fft(peaks[i]))))
        else: 
            max_height=np.max(np.real(np.fft.fftshift(np.fft.fft(peaks[i]))))
        
        ratio_temp=np.round(max_height/max_peak_height,decimals=2)
        ratio_collect.append(ratio_temp)
    return ratio_collect

#coupling pattern matrix 
def calc_couplings(omega,k):
    j_mat=np.zeros((k,k))#
    for j in range(0,k):
        for i in range(0,k):
            j_mat[i,j]=omega[j]-omega[i]
        
    j_mat=abs(np.round(pd.DataFrame(j_mat),decimals=1))
        
    return j_mat

# remove nans from bootstrap samples
def rm_nans(boot_samples):
     nan_idx=np.argwhere(np.isnan(np.array(boot_samples[0])))
     nan_idx=np.hstack(nan_idx)
     bootsamples_no_nan= boot_samples.drop(labels=nan_idx, axis=0)
     
     return bootsamples_no_nan
 

 
#################### functions not implemented yet####################################################
# remove outliers from bootstrap
#def rm_outliers():
 #   Q1=np.quantile(boot_samples['bootstrap_samples'][1],0.25)
  #  Q3=np.quantile(boot_samples['bootstrap_samples'][1],0.75)
  #  IQR=Q3-Q1
  #  cut1=Q1-IQR*1.5
   # cut2=Q3+IQR*1.5    
 #   idx=np.where( (cut1>test1[1]) &  (cut2<test1[1]) )

 
    