#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:38:29 2022

@author: mathies
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from helper_functions import ppm2hz
import nmrglue as ng

# make own SG filter

def virtual_echo(data):
     if data.ndim == 1:
        pts = data.size
        ve = np.zeros((2 * pts - 1), dtype="complex")
        ve[0] = np.real(data[0])
        ve[1:pts] = data[1:]
        ve[pts:] = data[1:][::-1].conj()
        return ve

def ROI_center(low_ppm,high_ppm,ve,fs,SF,O1p):
    # center of the ROI
    low_hz=ppm2hz(low_ppm,SF,O1p)
    high_hz=ppm2hz(high_ppm,SF,O1p)
    
    center=(low_hz+high_hz)/2
    
    # convert to indcies of the virtual echo
    center_idx=np.floor((ve.shape[0]-1)/(2*fs)*(fs+2*(0-center)))
    
    return center_idx

def filter_bandwidth(ve,low_ppm,high_ppm,SF,O1p,fs):
    N=len(ve)
    low_hz=ppm2hz(low_ppm,SF,O1p)
    high_hz=ppm2hz(high_ppm,SF,O1p)
    
    cut_low=np.floor((N-1)/(2*fs)*(fs+2*(0-low_hz)))
    cut_high=np.floor((N-1)/(2*fs)*(fs+2*(0-high_hz)))
    
    bandwidth=abs(cut_high-cut_low)
    
    return bandwidth

def filtering_fun(y,low_ppm,high_ppm,fs,SF,O1p):
    ve=virtual_echo(y)
    center=ROI_center(low_ppm,high_ppm,ve,fs,SF,O1p)
    bw=filter_bandwidth(ve, low_ppm, high_ppm, SF, O1p, fs)
    p=40.0
    
    for i, (n, c, b) in enumerate(zip(ve.shape, (center,), (bw,))):
            # 1D array of super-Gaussian for particular dimension
            if c is None:
                s = np.ones(n)
            else:
                s = np.exp(-(2 ** (p + 1)) * ((np.arange(1, n + 1) - c) / b) ** p)

            if i == 0:
                sg = s
            else:
                sg = sg[..., None] * s
            
            sg+(1j*np.zeros(sg.shape))

    spec=np.fft.fftshift(np.fft.fft(np.real(ve)-1j*np.imag(ve)))
    
    spec_filt=spec*sg
    
    return sg,spec,spec_filt,center

def noise_idx(y,noise_region,fs,SF,O1p):
      ve=virtual_echo(y)
      N=len(ve)
      low_hz=ppm2hz(noise_region[0],SF,O1p)
      high_hz=ppm2hz(noise_region[1],SF,O1p)
      
      noise_low=int(np.floor((N-1)/(2*fs)*(fs+2*(0-low_hz))))
      noise_high=int(np.floor((N-1)/(2*fs)*(fs+2*(0-high_hz))))
      
      return noise_low,noise_high
  



def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z    

def filter_noise(y,noise_region,spectrum,fs,SF,O1p):
    N=len(y)
    noise_list=[]
    
    slice_=slice(noise_idx(y,noise_region,fs,SF,O1p)[1],noise_idx(y,noise_region,fs,SF,O1p)[0])
    noise_region=spectrum[1][slice_]
    
    # get rid of baseline treds
    
    _, spectrum_als, info = baseline_arPLS(noise_region, lam=1e5, niter=20,
                                         full_output=True)
    
    # get more robust noise by having the averge level of 1000 realizations
    for i in range(0,1000):
        noise_als=np.random.normal(0,np.std(spectrum_als),N*2-1)
        noise_list.append(noise_als)
    
    noise_als=np.hstack(np.array([noise_list]))
    noise_als=np.mean(noise_als,axis=0)
    
    # convole noise and filter
    noise_als *= (1 -spectrum[0] )
    
    
    return noise_als

def onion_filter(low_ppm,high_ppm,noise_region,y,fs,SF,O1p):
    spectral_output=filtering_fun(y, low_ppm, high_ppm, fs, SF, O1p)
    syn_noise=filter_noise(y, noise_region, spectral_output, fs, SF, O1p)
    
    filtered_region=spectral_output[2]
    
    filtered_output=filtered_region+syn_noise
    
    slice_ = slice(0, filtered_output.shape[0] // 2) # remove the back half as data is a reversed symmetric echo
    FID_filtered_output = (2 * np.fft.ifft(np.fft.ifftshift( filtered_output)))[slice_]
    FID_filtered_output=np.real(FID_filtered_output)-1j*np.imag(FID_filtered_output) # make sure imaginary part turns the spectrum the correct way
    
    # make sure length of filtered output=length of filtered input (zero fill to one addtional point)
    FID_filtered_output=ng.proc_base.zf_size(FID_filtered_output, len(y))
    
    return FID_filtered_output


