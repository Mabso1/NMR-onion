#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:26:41 2022

@author: mathies
"""
import numpy as np
import pandas as pd
import math
import os


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

def result_csv(single_sinusoids,boot_samples,k,omega,SF,j_mat,O1,O1p):
    # amplitudes ratio
    k=len(omega)
    amp_ratios=amp_ratio(peaks=single_sinusoids, k=k)
    omega_ppm=hz2ppm(omega_hz=omega, SF=SF, O1p=O1p)

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
                             }))

    res_combo=pd.concat([res_region,res_uncertantinty,j_mat],axis=1)
   
    #Find the resultpath
    path=createResultPath("csv")
       
    path=path+"/results.csv"
   
    #save as csv
    res_combo.to_csv(path,index=True,sep=";")
    return res_combo

def createResultPath(filetype):
   
    #Diriger til result path og laver mappen hvis den ikke findes
    origpath=os.getcwd()+"/results"
    if not os.path.exists(origpath):
        os.makedirs(origpath)
   
    #Fortæller hvilke mapper der er i result mappen
    resultfiles=os.listdir(origpath)
   
    #Hvis mappen er tom, laver den den første og stopper funktionen
    if resultfiles==[]:
        path=origpath+"/ROI1"
        os.makedirs(path)
   
    #Hvis mappen ikke er tom
    else:
        #Den finder den nyeste mappe og går ind i den
        num=0
        for i in range(len(resultfiles)):
            end=len(resultfiles[i])
            newnum=int(resultfiles[i][3:end])
            if newnum>num:
                num=newnum
            else:
                continue
        testpath=origpath+"/ROI"+str(num)
        here=False
        testnames=os.listdir(testpath)
       
        #Så finder den om der er en fil af samme type inde i den nyeste mappe
        for i in range(len(testnames)):
            start=len(testnames[i])-len(filetype)
            end=len(testnames[i])
            if testnames[i][start:end]==filetype:
                here=True
            else:
                continue
       
        #Hvis der findes en fil så laver den en ny mappe, hvis ikke, gemmer den filen i den nyeste mappe
        if here==True:
            path=origpath+"/ROI"+str(num+1)
            os.makedirs(path)
        else:
            path=testpath
    return path
 
#################### functions not implemented yet####################################################
# remove outliers from bootstrap
#def rm_outliers():
 #   Q1=np.quantile(boot_samples['bootstrap_samples'][1],0.25)
  #  Q3=np.quantile(boot_samples['bootstrap_samples'][1],0.75)
  #  IQR=Q3-Q1
  #  cut1=Q1-IQR*1.5
   # cut2=Q3+IQR*1.5    
 #   idx=np.where( (cut1>test1[1]) &  (cut2<test1[1]) )

 
    