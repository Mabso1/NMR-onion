# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:27:50 2022

@author: mabso
"""
import numpy as np
import math
import scipy.optimize as optimization
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from helper_functions import ppm2hz,hz2ppm
from scipy import stats
from scipy import ndimage
import rle
import scipy.cluster.hierarchy as shc
import pandas as pd
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

#%%
def freq_hz (tn,fs):
    m=len(tn)
    fx=(tn-math.floor(m/2))*fs/(m-1)
    return(fx)

def ppm_omega(omega_hz,SF,O1p):
    omega_ppm=omega_hz/SF+O1p
    return(omega_ppm)

# recode rNMRfind
def time_calc(y, fs):
    N=len(y)
    tn=np.arange(0,N,1)
    t=tn*1/(fs)
    return t

def nls_fun(t,a,b):
    y=1-b*np.exp(-a*t)
    return y

def signal_decay(t,y):
   dat_re=np.real(y)
   y_sum=np.cumsum(abs(dat_re))
   y_sum=y_sum/max(y_sum)
   y_sum=y_sum[y_sum<0.99] 
   ones=np.repeat(1.0,len(y_sum))
   t=t[0:len(y_sum)]
   model_mat=np.vstack((ones,t)).T
   a0,a1=np.linalg.solve(model_mat.T@model_mat,model_mat.T@np.log(1-y_sum))
   
   start=np.array([a0,-a1])
   
   a,b=optimization.curve_fit(nls_fun, t, y_sum, start)
   
   return a0,a1,a

def estimate_n (y,width,height,fs):
    N=len(y)
    width=width/2
    x = width * np.sqrt((1 - height)/height)
    width=2*x
    tn=N/fs
    n=width*tn
    
    n=2*np.floor(n/2)+1
    
    return n

def apodize_sensitivity (t,R,y):
    N=len(y)
    f=np.exp(-R * t)
    out=np.fft.fft(y*f)/N
    return out


def apodize_resoultion (t,R,y):
    N=len(t)
    y_sum = np.cumsum(abs(y))
    y_sum = y_sum/max(y_sum)
    t_max = t[y_sum > 0.99][0]
    
    f = np.exp(R * t) * np.exp(-(R * t**2)/(2 * t_max))
    
    out=np.fft.fft(y*f)/N
    
    return out

def get_derivatives(y, n):

  # Forcing n to a minimum of 5, minimum to avoid interpolation
  if ( n < 5 ):
    # Disabling warning for now
    #warning("Increasing minimum filter width to avoid interpolation.")
    n = 5
# p=filter order 
# n=filter length
# m= order of derivative
  r1=signal.savgol_filter(np.real(y), window_length=n, polyorder=3,deriv=1)
  r2=signal.savgol_filter(np.real(y), window_length=n, polyorder=3,deriv=2)

  # Log filtering has issues with values on either side of zero
  r1 = abs(r1)
  r2=np.where(r2<0,-r2,0)

  #if(r2 < 0):
   #   r2=-r2
  #else:
   #   r2=0

  i1=signal.savgol_filter(np.imag(y), window_length=n, polyorder=3,deriv=1)
  i2=signal.savgol_filter(np.imag(y), window_length=n, polyorder=3,deriv=2)

  # Log filtering has issues with values on either side of zero
  i1=np.where(i1<0,-i1,0)
  
 # if(i1 < 0):
  #    i1=-i1
  #else:
   #   i1=0
  i2 =abs(i2)
  
  out=np.array([r1,r2,i1,i2]).T
  
  return out
  
def scaling_fun(y):
    scale=np.where(y<0,-np.log10(-y+1),np.log10(y+1))
    
    return scale

def find_PCA(y):
    standardizedData = StandardScaler(with_mean=True).fit_transform(y)

    pca=PCA(n_components=2)
    pca.fit(standardizedData)
    scores=pca.transform(standardizedData)
    PC1_score=scores[:,0]
    if ( abs(min(PC1_score)) > abs(max(PC1_score)) ):
        PC1_score = -PC1_score
    
    return PC1_score

def trim (roi,n):
    roi[0:n]=False
    roi[(len(roi) - n ):len(roi)] =False
    
    return roi

def open1(roi,n):
     k = np.repeat(1, (2 * math.floor(n/2) + 1))
     roi =np.array(roi,dtype=bool)
     y=ndimage.binary_opening(roi,k)
     return y

def dilate(roi,n):
    k = np.repeat(1, (2 * math.floor(n/2) + 1))
    y =np.array(roi,dtype=bool)
    y=ndimage.binary_dilation(y,k)
    return y

def list_regions(roi,ppm_val):
    start=np.where(np.diff(roi*1)==1)[0]+2#4
    stop=np.where(np.diff(roi*1)==-1)[0]+1
    start_val=ppm_val[start]
    stop_val=ppm_val[stop]
     
    return start_val, stop_val, start, stop

def get_rois(chemical_shift,intensity,fs,t,ppm_dil,peak_width,noise=4):
    
    index = np.flip(np.argsort(chemical_shift))
    chemical_shift =np.flip(chemical_shift[index])
    intensity = intensity[index]
    s=np.fft.ifft(intensity)
    
    R=signal_decay(t=t, y=s)
    R = R[2][0] * np.array([1.0, 0.75, 0.5])
    
    n = int(estimate_n(chemical_shift, peak_width, 0.5, fs))
    n_dil=ppm_dil/np.median(np.diff(chemical_shift))
    
    roi_list=[]
    
    for i in range(0,len(R)):
        ss=apodize_sensitivity(t=t, R=R[i], y=s)
        ds=get_derivatives(y=ss, n=n)
    
        pc1=find_PCA(y=ds)

        st_dev = stats.iqr(pc1)/1.35
        roi = pc1 > ( np.median(pc1) + noise * st_dev )
        roi = trim(roi, 100)
        roi = open1(roi, max(3, n))
        roi = dilate(roi, n_dil)
        
        roi_list.append(roi)
    
    values_hold=[]
    for i in range(0,len(roi_list)):
        values=sum(rle.encode(roi_list[0])[0])
        values_hold.append(values)
    
    idx=np.argmax(values_hold)
    roi=roi_list[idx]
    
    rois=list_regions(roi=roi,ppm_val=chemical_shift)
    
    return rois,roi

def find_roimax(roi,sr,ppm_val):
    xx=list_regions(roi,ppm_val)
    idxs_list=[]
    for i in range(0,len(xx[0])):
        xxa=np.arange(xx[2][i],xx[3][i])
        idx=np.append(xxa,xx[3][i])
        sr_max=np.argmax(np.real(sr[idx-1]))
        idxs=idx[sr_max]
        idxs_list.append(idxs)
    return idxs_list

def merge_peaks(temp_peaks,n):
    
    peak_sort=np.sort(temp_peaks)
    k=len(peak_sort)
    mat=peak_sort.reshape(k,1)
    
    Z = shc.single(squareform(abs(mat.T-mat)))
    cuts=np.hstack(shc.cut_tree(Z=Z,height=n))
    idx_types=np.unique(cuts)
    
    idx_list=[]
    peak_idx=[]
    for i in range(0,len(idx_types)):
        idx=np.where(cuts==idx_types[i])
        peak_sort_idx=peak_sort[idx]
        idx_list.append(idx)
        peak_idx.append(peak_sort_idx)
        
    merged_peaks=np.round(pd.DataFrame(peak_idx).mean(axis = 1).to_numpy())
    
    return merged_peaks

def peak_detection (ppm_val,y,width,noise_roi,noise_peaks,t,fs,stepsize=0.001):
    chemical_shift=ppm_val
    intensity=y
    index = np.flip(np.argsort(chemical_shift))
    chemical_shift =np.flip(chemical_shift[index])
    intensity = intensity[index]
    s=np.fft.ifft(intensity)
    
    R=signal_decay(t=t, y=s)
    R = R[2][0] * np.array([1.0, 0.75, 0.5])
    
    n = int(estimate_n(chemical_shift, width, 0.5, fs))
    peaks=[]
    
    roi_mask=get_rois(chemical_shift=ppm_val, intensity=y,fs=fs,t=t,ppm_dil=0.01,peak_width=width,noise=noise_roi)
    #roi_mask=roi_mask[1]
    
    for i in range(0,len(R)):
    #r = R[0]
        sr=apodize_resoultion(t=t, y=s, R=R[i])
        d = get_derivatives(y=sr, n=n)
        pc1=find_PCA(y=d)
    
        st_dev = stats.iqr(pc1)/1.35
        min_cutoff = np.median(pc1) + noise_peaks* st_dev
    
        min_cutoff = sum(pc1 < min_cutoff)/len(pc1)
    
        cutoff=1
    
        temp_peaks=[]
        
        while(True):
            cutoff=cutoff-stepsize
        #    print(cutoff,min_cutoff)
            proi=pc1>np.quantile(pc1,cutoff)
        
            proi=trim(proi,100)
            proi=open1(proi,max(3,n))
            proi=(proi) & (roi_mask[1])
    
    
            xx_max=find_roimax(roi=proi,sr=sr,ppm_val=ppm_val)
    
            temp_peaks=np.unique(np.append(np.asarray(temp_peaks),np.asarray(xx_max)))
    
    
            if (len(temp_peaks)<2):
                next
            
            else :
                temp_peaks=merge_peaks(temp_peaks=temp_peaks,n=n)
        
            if (cutoff<=min_cutoff):
                break
        peaks.append(temp_peaks)
        
    peak_lens=[]    
    for i in range(0,len(peaks)):
            peak_temp=len(peaks[i]) 
            peak_lens.append(peak_temp)
    idx=np.argmax(peak_lens)     
    x=peaks[idx].astype(int)
    
    detected_peaks=ppm_val[x]
    
    
    return detected_peaks


def peaks2hz(omega_ppm,high_ppm,low_ppm,SF,O1p):
    
    omega_hz=ppm2hz(ppm_value=omega_ppm, SF=SF, O1p=O1p) # conver to hz
    # make sure that peaks are only within the region and save as omega_hz_filtered
    idx=np.where( (-omega_hz>ppm2hz(low_ppm,SF,O1p)) &  (-omega_hz<ppm2hz(high_ppm,SF,O1p)) )
    omega_hz_filtered=-omega_hz[idx]
    
    return omega_hz_filtered

def signal_points(omega_hz_filtered,freq,y,SF,O1p,ppm_out):
    hz_points=[]
    signal_points=[]
    for i in range(0,len(omega_hz_filtered)):
        idx=np.argmin(abs(freq-omega_hz_filtered[i]))
        hz_point=freq[idx]
        signal_point=np.fft.fftshift(np.fft.fft(y))[idx]
        hz_points.append(hz_point)
        signal_points.append(signal_point)
        
        if (ppm_out==True):
            hz_points=hz2ppm(omega_hz=hz_points, SF=SF, O1p=O1p)
        
    return hz_points,signal_points


from scipy.signal import find_peaks, argrelextrema
from scipy.integrate import trapz

def auto_integrate_peaks(x, y, known_peaks, rel_height=0.05):
    """
    Integrates peaks in a spectrum using known peak x-positions.

    Parameters:
        x (np.ndarray): x-axis values (e.g. ppm)
        y (np.ndarray): y-axis values (e.g. intensity or |FFT| magnitude)
        known_peaks (list or np.ndarray): list of known peak x-positions
        rel_height (float): height ratio (0–1) to define peak width for integration

    Returns:
        areas (list): integrated areas under each peak
        bounds (list): list of (left_x, right_x) tuples per peak
        peaks_x (list): actual x-values of peak centers
    """
    x = np.array(x)
    y = np.array(y)
    known_peaks = np.array(known_peaks)

    # Ensure y is magnitude for integration
    y = np.abs(y)

    # Get local minima to help define bounds
    minima_idxs = argrelextrema(y, np.less)[0]

    areas = []
    bounds = []
    peaks_x = []

    for peak_pos in known_peaks:
        peak_idx = np.argmin(np.abs(x - peak_pos))
        peak_val = y[peak_idx]
        threshold = peak_val * rel_height

        # Find left bound
        left = peak_idx
        while left > 0 and y[left] > threshold:
            left -= 1
        # Optionally snap to local minimum
        left_mins = minima_idxs[minima_idxs < peak_idx]
        if len(left_mins) > 0:
            left = max(left, left_mins[-1])

        # Find right bound
        right = peak_idx
        while right < len(y) - 1 and y[right] > threshold:
            right += 1
        # Optionally snap to local minimum
        right_mins = minima_idxs[minima_idxs > peak_idx]
        if len(right_mins) > 0:
            right = min(right, right_mins[0])

        # Extract region
        x_region = x[left:right+1]
        y_region = y[left:right+1]

        if len(x_region) < 2:
            continue  # Skip if the region is too narrow

        area = trapz(y_region, x_region)

        areas.append(area)
        bounds.append((x[left], x[right]))
       # peaks_x.append(x[peak_idx])

    return areas, bounds

import os


import os
import matplotlib.pyplot as plt
import numpy as np

def integrate_peaks(xlim1_ppm, xlim2_ppm, ylim1, ylim2, single_sinusoids,
                    ppm_val, par_res, SF, O1p, rel_height=0.15, plot=True,
                    save=False, save_dir='.', filename='integration_plot', format='png'):
    int_list = []
    omega_list = []
    omega_ppm = hz2ppm(par_res['omega_est'], SF, O1p)

    k = len(single_sinusoids)  # Make sure 'k' is defined

    if plot:
        plt.figure(figsize=(10, 6))  # Create a single figure

    for i in range(k):
        y0_mag = np.real(np.fft.fftshift(np.fft.fft(single_sinusoids[i])))
        areas, bounds = auto_integrate_peaks(ppm_val, y0_mag, known_peaks=[omega_ppm[i]], rel_height=rel_height)
        peaks_x = ppm_val[np.where(y0_mag == np.max(y0_mag))[0]]

        int_list.append(areas)
        omega_list.append(peaks_x)

        if plot:
            plt.plot(ppm_val, y0_mag)
            for j, (left, right) in enumerate(bounds):
                plt.axvspan(left, right, color='orange', alpha=0.3,
                            label='Integration region' if i == 0 and j == 0 else None)
            plt.scatter(peaks_x, np.interp(peaks_x, ppm_val, y0_mag), color='red',
                        label='Detected peaks' if i == 0 else None)

    if plot:
        plt.xlabel('ppm')
        plt.ylabel('Intensity')
        plt.xlim(xlim1_ppm, xlim2_ppm)
        plt.ylim(ylim1, ylim2)
        plt.title('FFT Spectra with Auto Integration')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()

        if save:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'{filename}.{format}')
            plt.savefig(path, dpi=300, format=format)
            print(f"Plot saved to: {path}")

        plt.show()

    int_list = np.hstack(np.array(int_list))
    omega_hz = ppm2hz(np.hstack(omega_list), SF, O1p)
    omega_list = np.hstack(np.array(omega_list))

    return omega_list, omega_hz, int_list

def onion_peak_detection(width, noise_peaks, cut, ylim1, ylim2, y_fft_filt, y_filt,
                         low_ppm, high_ppm, noise_level, ppm_val, freq, t, fs, SF, O1p, plot=True):

    # 1. Detect peaks in ppm
    omega_ppm = peak_detection(ppm_val=ppm_val, y=y_fft_filt, width=width,
                               noise_roi=0, noise_peaks=noise_peaks, t=t, fs=fs)

    # 2. Convert detected peaks to Hz and filter within ROI
    omega_hz_filtered = peaks2hz(omega_ppm, high_ppm, low_ppm, SF, O1p)

    # 3. Get signal points (Hz and Intensity)
    ROI_hz_points, ROI_signal_points = signal_points(omega_hz_filtered=omega_hz_filtered,
                                                     freq=freq, y=y_filt, SF=SF, O1p=O1p, ppm_out=False)
    ROI_ppm_points = hz2ppm(np.array(ROI_hz_points), SF, O1p)

    # 4. Compute SNR in dB
    SNR_db = np.log10(np.real(ROI_signal_points) / np.std(noise_level))
    idx = np.where(SNR_db > 0)[0]

    ROI_signal_points = np.array(ROI_signal_points)[idx]
    ROI_ppm_points = np.array(ROI_ppm_points)[idx]
    omega_hz_filtered = np.array(omega_hz_filtered)[idx]

    # 5. Apply intensity-based cut if needed
    if cut > 0:
        sorted_vals = np.sort(np.real(ROI_signal_points))
        trimmed_vals = sorted_vals[cut:]
        threshold = np.min(trimmed_vals)

        print(f"Min value found: {np.min(np.real(ROI_signal_points))}, threshold set at: {threshold}")

        keep_mask = np.real(ROI_signal_points) >= threshold
        reject_mask = ~keep_mask
        

        ROI_signal_points_kept = ROI_signal_points[keep_mask]
        ROI_ppm_points_kept = ROI_ppm_points[keep_mask]
        omega_hz_filtered = omega_hz_filtered[keep_mask]

        ROI_signal_points_cut = ROI_signal_points[reject_mask]
        ROI_ppm_points_cut = ROI_ppm_points[reject_mask]

    else:
        print(f"No cutoff applied (cut=0). Min value: {np.min(np.real(ROI_signal_points))}")
        ROI_signal_points_kept = ROI_signal_points
        ROI_ppm_points_kept = ROI_ppm_points
        ROI_signal_points_cut = np.array([])
        ROI_ppm_points_cut = np.array([])

    # 6. Plotting
    if plot:
        plt.figure(figsize=(8, 4))
        plt.xlabel('ppm')
        plt.ylabel('Intensity')
        plt.title(f"Peaks detected ({low_ppm}-{high_ppm} ppm)")
        plt.plot(ppm_val, y_fft_filt, color="blue")
        plt.xlim(low_ppm, high_ppm)
        plt.ylim(ylim1, ylim2)

        plt.scatter(ROI_ppm_points_kept, ROI_signal_points_kept, color="red", label="Kept peaks")
        if len(ROI_ppm_points_cut) > 0:
            plt.scatter(ROI_ppm_points_cut, ROI_signal_points_cut, color="purple", label="Cut-off peaks")

        plt.axhline(np.min(np.real(ROI_signal_points[keep_mask])), color="black", linestyle="--", label="Min retained")
        plt.legend()

    return omega_hz_filtered
