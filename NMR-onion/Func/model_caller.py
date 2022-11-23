#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:14:48 2022

@author: mathies
"""
import sys
import time
path="/home/mathies/Desktop/nmr onion/NMR onion final"
sys.path.append(path)


def onion_model_call(model_name,omega_hz_filtered,tn_new,t_new,fs_new,y_norm):
    if (model_name=="skewed_lorentzian"):
        from torch_loss_fn_full_freq_lor import onion_lor
        
        start =time.time()
        model_fit=onion_lor(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_norm)
        print((time.time() - start))
        
        return model_fit
    
    elif(model_name=="skewed_genvoigt"):
        from torch_loss_fn_full_freq_gen_voigt import onion_genvoigt
        
        start =time.time()
        model_fit=onion_genvoigt(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_norm)
        print((time.time() - start))
        
        return model_fit
        
    elif(model_name=="skewed_pvoigt"):
        from torch_loss_fn_full_freq_p_voigt import onion_pvoigt
        
        start =time.time()
        model_fit=onion_pvoigt(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_norm)
        print((time.time() - start))
    
        return model_fit
    
    elif(model_name=="pvoigt"):
        from torch_loss_fn_full_freq_p_voigt_nonskew import onion_pvoigt_noskew
        
        start =time.time()
        model_fit=onion_pvoigt_noskew(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_norm)
        print((time.time() - start))
        
        return model_fit
    
    elif(model_name=="genvoigt"):
        from torch_loss_fn_full_freq_gen_voigt_nonskew import onion_noskew_genvoigt
        
        start =time.time()
        model_fit=onion_noskew_genvoigt(omega=omega_hz_filtered, tn=tn_new, t=t_new, fs=fs_new, y=y_norm)
        print((time.time() - start))
        
        return model_fit
    
    
def onion_bootstrap_call(B,CI_level,alpha,omega,scale1,scale2,t,k,y,fs,model_name,low_ppm,high_ppm,freq,SF,O1p,eta=None):
    if (model_name=="skewed_lorentzian"):
        from lorentzian_bootstrap_function import lorentzian_wild_boostrap
        
        start =time.time()
        boot=lorentzian_wild_boostrap(B,CI_level,alpha,omega,scale1,scale2,t,k,y,fs,model_name,low_ppm,high_ppm,freq,SF,O1p)
        print((time.time() - start))
        
        return boot
    
    elif(model_name=="skewed_pvoigt"):
        from pvoigt_bootstrap_function import pvoigt_wild_boostrap
        
        start =time.time()
        boot=pvoigt_wild_boostrap(B,CI_level,alpha,omega,eta,scale1,scale2,t,k,y,fs,model_name,low_ppm,high_ppm,freq,SF,O1p)
        print((time.time() - start))
        
        return boot
        
    elif(model_name=="skewed_genvoigt"):
        from genvoigt_bootstrap_function import genvoigt_wild_boostrap
        
        start =time.time()
        boot=genvoigt_wild_boostrap(B,CI_level,alpha,omega,eta,scale1,scale2,t,k,y,fs,model_name,low_ppm,high_ppm,freq,SF,O1p)
        print((time.time() - start))
        
    
        return boot