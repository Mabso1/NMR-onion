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
import matplotlib.pyplot as plt


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

# get raw amplitude heights of real spectrum
def get_amps(peaks,k):
    peaks_collect=[]
    for i in range(0,k):
        max_peak=np.real(np.max(np.fft.fftshift(np.fft.fft(peaks[i]))))
        peaks_collect.append(max_peak)
    return peaks_collect

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

# count number of overlaps in ROI
# returns 0=no overlap, 1=overlap
def count_overlaps(idx_list,k):
    dummy_array=np.repeat(np.nan,k)
    dummy_array[idx_list]=1
    dummy_array[np.isnan(dummy_array)]=0
    
    return dummy_array

# remove nans from bootstrap samples
def rm_nans(boot_samples):
     nan_idx=np.argwhere(np.isnan(np.array(boot_samples[0])))
     nan_idx=np.hstack(nan_idx)
     bootsamples_no_nan= boot_samples.drop(labels=nan_idx, axis=0)
     
     return bootsamples_no_nan

def result_csv(single_sinusoids,boot_samples,k,omega,idx_list,SF,j_mat,O1,O1p):
    # amplitudes ratio
    k=len(omega)
    amp_ratios=amp_ratio(peaks=single_sinusoids, k=k)
    raw_amps=get_amps(peaks=single_sinusoids, k=k)
    omega_ppm=hz2ppm(omega_hz=omega, SF=SF, O1p=O1p)
    
    overlap_array=count_overlaps(idx_list, k)
    
    # get correct indexing
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
                             'overlaps':overlap_array,
                             'amp_height': raw_amps
                             }))

    res_combo=pd.concat([res_region,res_uncertantinty,j_mat],axis=1)
   
    #Find the resultpath
    path=createResultPath("csv")
       
    path=path+"/results.csv"
   
    #save as csv
    res_combo.to_csv(path,index=True,sep=";")
    return res_combo


def result_csv_noboot(single_sinusoids,k,omega,SF,j_mat,O1,O1p):
    # amplitudes ratio
    k=len(omega)
    amp_ratios=amp_ratio(peaks=single_sinusoids, k=k)
    raw_amps=get_amps(peaks=single_sinusoids, k=k)
    omega_ppm=hz2ppm(omega_hz=omega, SF=SF, O1p=O1p)
    


    # table of regional results
    res_region=pd.DataFrame(({'amp_ratio':amp_ratios,
                             'omega_hz':omega+O1,
                             'omega_ppm':omega_ppm,
                             'amp_height': raw_amps
                             }))

    res_combo=pd.concat([res_region,j_mat],axis=1)
   
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
 
def getoverlaps(boot_samples,omega_ppm,k):
    
    # get omega bootstrap samples and index according to number of sinusoid k
    omega_samples=boot_samples['omega_boot']
    omega_samples.index = range(k)
    
    # get CI uppper and lower for the bootraps and index accroding to number of k sinusoids    
    roi_CIl=boot_samples['omega_CI_lower']
    roi_CIu=boot_samples['omega_CI_upper']

    roi_CIl.index = range(k)
    roi_CIu.index = range(k)
    
    # collect CIs in dataframe
    df=pd.concat((roi_CIl,roi_CIu),axis=1)
    df.columns=np.array(['omega_CI_lower','omega_CI_upper'])
    
    # find overlapping CIs
    df = df.sort_values("omega_CI_lower").reset_index(drop=True)
    idx = 0
    dfs = []
    while True:
        low = df.omega_CI_lower[idx]
        high = df.omega_CI_upper[idx]
        sub_df = df[(df.omega_CI_lower <= high) & (low <= df.omega_CI_upper)]
        dfs.append(sub_df)
        idx = sub_df.index.max() + 1
        if idx > df.index.max():
            break
    # set the collected overlappings CIs as array
    dummy=[]
    for i in range(0,len(dfs)):
        grps=np.array([dfs[i].index])
        dummy.append(grps)
  
    
    overlap_list=[]
    for i in range(0,len(dummy)):
        omega_ppm_list=omega_ppm[dummy[i]]
        overlap_list.append(omega_ppm_list)    
   
    # replace non overlaps with nans
    for i in range(0,len(dummy)):
        bool_point=len(overlap_list[i][0])>1
    
        if(bool_point==True):
            overlap_list[i][0]=overlap_list[i][0]
        else:
            overlap_list[i][0]=np.nan
    # find the index of non nans
    overlap_dots=np.hstack(np.hstack(overlap_list))
    idx_overlaps=np.argwhere(~np.isnan(overlap_dots))
    
    test=len(idx_overlaps)==0
    
    if (test==True):
        return []
    
    overlap_dots=overlap_dots[idx_overlaps]
   
    # collect the index list of overlaps
    idx_list=[]
    for i in range(0,len(overlap_dots)):
        idx_point=np.argwhere(overlap_dots[i]==omega_ppm)
        idx_list.append(idx_point)
    idx_list=np.hstack(np.hstack(idx_list))
    
    return idx_list

# plot histogram of up to 3 peaks next to one another to check for overlaps
def plot_freq_CIs(boot_samples,k,O1,idx_list,peak1,peak2,peak3=None,saveplot=False,myplot_name=None):
    
    # get bootstrap samples
    omega_samples=boot_samples['omega_boot']
    omega_samples.index = range(k)
    
    # set min and max for histograms
    df_max=omega_samples.T.max(axis=1).max()+O1
    df_min=omega_samples.T.min(axis=1).min()+O1
    
    # get CI cutoffs
    sub_roi_CIl=boot_samples['omega_CI_lower']+O1
    sub_roi_CIu=boot_samples['omega_CI_upper']+O1

    sub_roi_CIl.index = range(k)
    sub_roi_CIu.index = range(k)
    
    # get up to three peak bootstaps to plot
    #hz_range=ppm2hz(ppm_range,SF,O1p)
    peak1_idx=np.where(peak1==idx_list)
    peak2_idx=np.where(peak2==idx_list)
    peak3_idx=np.where(peak3==idx_list)
    
    # set range of plot
    range_min=omega_samples.T[idx_list[peak1_idx]].min(axis=1).min()+O1-3
    
    if (peak3==None):
        range_max=omega_samples.T[idx_list[peak2_idx]].max(axis=1).max()+O1+3
   
    else:
        range_max=omega_samples.T[idx_list[peak3_idx]].max(axis=1).max()+O1+3
    
    _, bins, _ = plt.hist(omega_samples.T[idx_list[0]], bins=500,range=(df_min,df_max))
    plt.title("boostrap samples of peaks")
    plt.xlabel("Hz")
    plt.ylabel("Emperical density")
  
    plt.xlim(range_min,range_max)
    #PFP_len=len(idx_list)
    #for i in range(0,PFP_len):
    _= plt.hist(omega_samples.T[idx_list[peak1_idx]]+O1,color="r", bins=bins,density=True)  
    _= plt.hist(omega_samples.T[idx_list[peak2_idx]]+O1,ec="b",fc="none", bins=bins,density=True)    
    _= plt.hist(omega_samples.T[idx_list[peak3_idx]]+O1,color="r", bins=bins,density=True)  
    plt.axvline(sub_roi_CIl[idx_list[np.hstack(peak1_idx)[0]]],color="green")
    plt.axvline(sub_roi_CIu[idx_list[np.hstack(peak1_idx)[0]]],color="green")
    plt.axvline(sub_roi_CIl[idx_list[np.hstack(peak2_idx)[0]]],color="black")
    plt.axvline(sub_roi_CIu[idx_list[np.hstack(peak2_idx)[0]]],color="black")
    plt.axvline(sub_roi_CIl[idx_list[np.hstack(peak3_idx)[0]]],color="yellow")
    plt.axvline(sub_roi_CIu[idx_list[np.hstack(peak3_idx)[0]]],color="yellow")
    
    # save the plot, default is false
    if (saveplot==True):
        plt.savefig('myplot_name')



#################### functions not implemented yet####################################################
## automatic generation of all bootrap samples histrogram plot functiom

#df_max=omega_samples.T.max(axis=1).max()+O1
#df_min=omega_samples.T.min(axis=1).min()+O1

#sub_roi_CIl=res_combo['omega_hz_CI_lower']
#sub_roi_CIu=res_combo['omega_hz_CI_upper']

#sub_roi_CIl.index = range(k)
#sub_roi_CIu.index = range(k)

#df=pd.concat((sub_roi_CIl,sub_roi_CIu),axis=1)

#_, bins, _ = plt.hist(omega_samples.T[idx_list[0]], bins=500,range=(df_min,df_max))
#plt.title("boostrap samples of two peaks")
#plt.xlabel("Hz")
#plt.ylabel("Emperical density")
#plt.xlim(-2340+O1,-2350+O1)
#for i in range(0,3):
#    _= plt.hist(omega_samples.T[idx_list[i+13]]+O1,fc="none",ec="b", bins=bins)
#    plt.axvline(sub_roi_CIl[idx_list[i+13]])
#    plt.axvline(sub_roi_CIu[idx_list[i+13]])
#plt.savefig('peaks_overlaps_subROI1_figure1.svg')  
    