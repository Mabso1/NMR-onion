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
from helper_functions import ppm2hz,freq_hz,t_disrecte,time_series,ppm_axis1
import scipy.signal as signal
from data_import import import_data
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


def filter_bandwidth_hz(ve,low_ppm,high_ppm,SF,O1p,fs):
  #  N=len(ve)
    low_hz=ppm2hz(low_ppm,SF,O1p)
    high_hz=ppm2hz(high_ppm,SF,O1p)
    
    #cut_low=np.floor((N-1)/(2*fs)*(fs+2*(0-low_hz)))
    #cut_high=np.floor((N-1)/(2*fs)*(fs+2*(0-high_hz)))
    
    bandwidth=abs(high_hz-low_hz)
    
    return bandwidth

def filter_bandwidth_hz_idx(ve,low_hz,high_hz,fs):
    N=len(ve)
    #low_hz=ppm2hz(low_ppm,SF,O1p)
    #high_hz=ppm2hz(high_ppm,SF,O1p)
    
    cut_low=np.floor((N-1)/(2*fs)*(fs+2*(0-low_hz)))
    cut_high=np.floor((N-1)/(2*fs)*(fs+2*(0-high_hz)))
    
    bandwidth=abs(cut_high-cut_low)
    
    return bandwidth


def filtering_fun_ajusting_p(y,low_ppm,high_ppm,fs,SF,O1p,p):
    ve=virtual_echo(y)
    center=ROI_center(low_ppm,high_ppm,ve,fs,SF,O1p)
    bw=filter_bandwidth(ve, low_ppm, high_ppm, SF, O1p, fs)
    
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
    np.random.seed(3)
    N=len(y)
    noise_list=[]
   # noise_list_mean=[]
    
    slice_=slice(noise_idx(y,noise_region,fs,SF,O1p)[1],noise_idx(y,noise_region,fs,SF,O1p)[0])
    noise_region=spectrum[1][slice_]
    
    # get rid of baseline treds
    
    _, spectrum_als, info = baseline_arPLS(noise_region, lam=1e5, niter=20,
                                         full_output=True)
   # noise_als=np.random.normal(0,np.std(spectrum_als),N*2-1)
    # get more robust noise by having the averge level of 1000 realizations
    for i in range(0,1000):
        noise_als=np.random.normal(0,np.std(spectrum_als),N*2-1)
        noise_list.append(noise_als)
    
    noise_level=np.mean(np.std(noise_list,axis=1))
   # for i in range(0,1000):
    noise_als=np.random.normal(0,noise_level,N*2-1)
     #   noise_list_mean.append(noise_als)
    
    #noise_als=np.median(noise_list,axis=0)
    noise_out=noise_als
    
    # convole noise and filter
    noise_als *= (1 -spectrum[0] )
    
    
    return noise_als,noise_out,noise_region


def onion_filter(low_ppm,high_ppm,noise_region,y,fs,SF,O1p):
    spectral_output=filtering_fun(y, low_ppm, high_ppm, fs, SF, O1p)
    syn_noise=filter_noise(y, noise_region, spectral_output, fs, SF, O1p)
    
    syn_noise_level=syn_noise[1]
    
    filtered_region=spectral_output[2]
    
    filtered_output=filtered_region+syn_noise[0]
    
    slice_ = slice(0, filtered_output.shape[0] // 2) # remove the back half as data is a reversed symmetric echo
    FID_filtered_output = (2 * np.fft.ifft(np.fft.ifftshift( filtered_output)))[slice_]
    FID_filtered_output=np.real(FID_filtered_output)-1j*np.imag(FID_filtered_output) # make sure imaginary part turns the spectrum the correct way
    
    # make sure length of filtered output=length of filtered input (zero fill to one addtional point)
    FID_filtered_output=ng.proc_base.zf_size(FID_filtered_output, len(y))
    
    return FID_filtered_output,syn_noise_level

#fouriershift theorem implementation
def fourier_shift(N,shift,y):
    N=len(y)
    if (N%2==0):
        kma=-N/2
        kmi=N/(2-1)
        
        kk=np.arange(kma,kmi)
        k_idx1=np.hstack(np.arange(N/2,N)).astype(int)
        k_idx2=np.hstack(np.arange(0,N/2)).astype(int)
        ee=np.concatenate((k_idx1,k_idx2))
        kk=kk[ee]
        ff=2*np.pi*kk/(N)

        y_shift=(y*np.exp(-1j*shift*ff))
    else:
        kma=-(N-1)/2
        kmi=(N-1)/2
        
        kk=np.arange(kma,kmi)
        k_idx1=np.hstack(np.arange(N/2,N-1)).astype(int)
        k_idx2=np.hstack(np.arange(0,N/2)).astype(int)
        ee=np.concatenate((k_idx1,k_idx2))
        kk=kk[ee]
        ff=2*np.pi*kk/(N)
   
        y_shift=(y*np.exp(-1j*shift*ff))
    return(y_shift)


# overmid shifts

def shift_overmid(high_cut,low_cut,freq,y,moveback): 
    N=len(y)
    # note minimum bandwidth=492
    bandwidth=high_cut-low_cut # find the bandwidth
    move_bottom=np.argmin(abs(freq-low_cut)) # move frequies with anker point at lower end
    move_0=np.argmin(abs(freq-0)) # move index corresponding of an fftshift
    f_temp=np.argmin(abs(freq-bandwidth/2))
    y_shifted=fourier_shift(N=N,shift=move_bottom-move_0,y=y)
    y_shifted_overmid=fourier_shift(N=N,shift=f_temp-move_0,y=y_shifted)
    
    hz_shifted=-freq[f_temp]-freq[move_bottom]-freq[move_0]
    
    if (moveback==True):
         y_shifted_overmid=fourier_shift(N=N,shift=-f_temp-move_bottom+move_0*2,y=y_shifted_overmid)
    
    return(y_shifted_overmid,hz_shifted)

def signal_decimate(y_filt,low_ppm,high_ppm,freq,SF,O1p,fs):
    ve=virtual_echo(y_filt)
    bw_hz=filter_bandwidth_hz(ve, low_ppm, high_ppm, SF, O1p, fs)
    
    q_values=np.arange(1,1000)
    
    counts=np.where((bw_hz/2)<fs*0.5/q_values)[0]
    
    q_max=len(counts)
    
    q_max=np.min(np.array([30,q_max]))
    #if (q_max>=30):
     #   q_max=q_max-5
    #else:
     #   q_max=q_max
    
    high_hz=ppm2hz(high_ppm,SF,O1p)
    low_hz=ppm2hz(low_ppm,SF,O1p)
    
    y_centered=shift_overmid(high_hz, low_hz, freq, y_filt, False)
    
    hz_moved=y_centered[1]
    
    y_dem = signal.decimate(y_centered[0], q=q_max,ftype="fir")

    fs_new=fs/q_max
    tn_new=np.arange(0,len(y_dem),1)
    t_new=1/(fs_new)*tn_new
    freq_new=freq_hz(tn_new, fs_new)
    
    return y_dem,fs_new,tn_new,t_new,freq_new,hz_moved



def data2fit(low_ppm,high_ppm,noise_region,data,fs,O1,fs_ppm,SF,O1p,ZF):

    idx_zero=np.where(np.real(data)==0)
    data=np.delete(data,idx_zero)
    if (ZF==True):
        data = ng.proc_base.zf_size(data, len(data)*2,mid=False)
    else:
        data=data
        
    N=len(data)

    # define discrete time (tn) and measure timed (t)
    tn=t_disrecte(N)
    t=time_series(tn=tn,fs=fs)

    # define ppm axis
    freq=freq_hz(tn=tn, fs=fs)
    ppm_val=ppm_axis1(time=t, O1=O1, fs=fs, fs_ppm=fs_ppm)

    # apply digital filter to get region of interest and estimated noise level
    y_filt,noise_level=onion_filter(low_ppm, high_ppm, noise_region, data, fs, SF, O1p)

    # fourier transform filtered data
    y_fft_filt=np.fft.fftshift(np.fft.fft((y_filt)))
   
    return y_fft_filt,y_filt,freq,ppm_val,t,freq,tn
        

