#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:19:21 2022

@author: mathies
"""
import numpy as np
import nmrglue as ng


# import data function
# defines how to import data from processed data file
### inputs ###
# processed data directory 
### returns ###
# 1 processed data fid
# 2 data length
# 3 sample rate (sweep width in hz)
# 4 offset (O1 in hz)
# 5 spectrometor frequency 
# 6 sample rate in ppm (sweep width)
# 7 O1p scaled ppm O1 value
def import_data(path,zerofill):
    dic, data = ng.bruker.read_pdata(path) # read the data and parameters 
    slice_ = slice(0, data.shape[0] // 2) # remove the back half as data is a reversed symmetric echo
    data = (2 * np.fft.ifft(np.fft.ifftshift(data)))[slice_] # remove the backhalf of singal and time the front end by 2
    
    data=np.real(data)-1j*np.imag(data) # make sure imaginary part turns the spectrum the correct way
    
    if (zerofill==True):
        data = ng.proc_base.zf_size(data, len(data)*2**1,mid=False) # zero fill the data back to orignal length
       # data=ng.proc_base.zf_auto(data)
        N=len(data)
    else:
        data=data
        N=len(data)
        
    fs=dic['acqus']['SW_h'] # sample rate/sweep width
    O1=dic['acqus']['O1'] # O1 (transmitter poisition)
    SF=dic['procs']['SF'] # spectrometor frequency
    fs_ppm=dic['acqus']['SW'] # sample rate/sweep width in ppm
    O1p=O1/SF # transmitter poistion in ppm
    
    return data, N, fs, O1, SF, fs_ppm, O1p





