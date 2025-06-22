#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:58:14 2025

@author: mabso
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import getoverlaps
# this contains the plotting functions of NMR-onion

def plot_results(xlim1_ppm, xlim2_ppm, ylim1, ylim2, xlim1, xlim2,
                 y_hat, y_filt, t, freq, ppm_val, k, single_sinusoids,
                 time_domain=True, freq_domain_ppm=True, freq_domain=False,
                 residuals=False, show_peaks=True, save_dir=None,
                 save_format='png'):
    
    def save_figure(fig, name):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{name}.{save_format}"
            fig_path = os.path.join(save_dir, filename)
            fig.savefig(fig_path, dpi=300, format=save_format, bbox_inches='tight')
            print(f"Saved: {fig_path}")

    # --- Time domain plot ---
    if time_domain:
        y_filt_norm = y_filt / np.linalg.norm(y_filt)
        residual = y_filt_norm - y_hat

        fig1 = plt.figure(figsize=(10, 4))
        plt.plot(t, y_filt_norm, color="blue", label='Observed FID (normalized)')
        plt.plot(t, y_hat, color="black", label='Predicted FID')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Time-Domain FID vs Model")
        plt.legend()
        plt.tight_layout()
        save_figure(fig1, "time_domain_fid_vs_model")
        plt.show()

        if residuals:
            fig2 = plt.figure(figsize=(10, 3))
            plt.plot(t, residual, color="orange")
            plt.xlabel("Time")
            plt.ylabel("Residual Amplitude")
            plt.title("Time-Domain FID Residuals (Observed - Predicted)")
            plt.tight_layout()
            save_figure(fig2, "time_domain_residuals")
            plt.show()

    # --- Frequency-domain (Hz) ---
    y_fft_hat = np.fft.fftshift(np.fft.fft(y_hat))
    y_fft_cut = np.fft.fftshift(np.fft.fft(y_filt / np.linalg.norm(y_filt)))
    y_fft_resi = y_fft_cut - y_fft_hat

    if freq_domain:
        fig3 = plt.figure(figsize=(10, 4))
        for i in range(k):
            y0_mag = np.real(np.fft.fftshift(np.fft.fft(single_sinusoids[i])))
            peaks_x = freq[np.where(y0_mag == np.max(y0_mag))[0]]
            plt.plot(freq, y0_mag, color="red", alpha=0.5, label='Component FFT' if i == 0 else None)
            if show_peaks:
                plt.scatter(peaks_x, np.interp(peaks_x, freq, y0_mag),
                            color='purple', label='Detected peaks' if i == 0 else None)

        plt.plot(freq, y_fft_cut, color="blue", label="Observed FFT")
        plt.plot(freq, y_fft_hat, color="black", label="Predicted FFT")
        if residuals:
            plt.plot(freq, y_fft_resi, color="orange", label="FFT Residual")

        plt.xlim(xlim1, xlim2)
        plt.ylim(ylim1, ylim2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Intensity")
        plt.title("Frequency-Domain (Hz)")
        plt.legend()
        plt.tight_layout()
        save_figure(fig3, "frequency_domain_hz")
        plt.show()

    # --- Frequency-domain (ppm) ---
    if freq_domain_ppm:
        fig4 = plt.figure(figsize=(10, 4))
        for i in range(k):
            y0_mag = np.real(np.fft.fftshift(np.fft.fft(single_sinusoids[i])))
            peaks_x = ppm_val[np.where(y0_mag == np.max(y0_mag))[0]]
            plt.plot(ppm_val, y0_mag, color="red", alpha=0.5, label='Component FFT' if i == 0 else None)
            if show_peaks:
                plt.scatter(peaks_x, np.interp(peaks_x, ppm_val, y0_mag),
                            color='purple', label='Detected peaks' if i == 0 else None)

        plt.plot(ppm_val, y_fft_cut, color="blue", label="Observed FFT")
        plt.plot(ppm_val, y_fft_hat, color="black", label="Predicted FFT")
        if residuals:
            plt.plot(ppm_val, y_fft_resi, color="orange", label="FFT Residual")

        plt.xlim(xlim1_ppm, xlim2_ppm)
        plt.ylim(ylim1, ylim2)
        plt.xlabel("ppm")
        plt.ylabel("Intensity")
        plt.title(f"Frequency-Domain (ppm): {xlim1_ppm}–{xlim2_ppm}")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        save_figure(fig4, "frequency_domain_ppm")
        plt.show()
        
def plot_model_bootstrap(ppm_val, single_sinusoids, y_hat, y_filt,
                         xlim1_ppm, xlim2_ppm, ylim1, ylim2,
                         k, boot_samples=None, omega_ppm=None,
                         show_residuals=True, show_bootstrap_peaks=True,
                         save_dir=None, save_format='png', filename_prefix='model_bootstrap'):
    """
    Plots FFT model results in ppm domain with optional bootstrap peaks and saves the figure.

    Parameters:
    - save_dir: directory to save plot (if None, won't save)
    - save_format: 'png', 'pdf', etc.
    - filename_prefix: name prefix for saved file
    """

    # Compute FFTs
    y_fft_hat = np.fft.fftshift(np.fft.fft(y_hat))
    y_fft_cut = np.fft.fftshift(np.fft.fft(y_filt / np.linalg.norm(y_filt)))
    y_fft_resi = y_fft_cut - y_fft_hat

    # Create figure
    fig = plt.figure(figsize=(10, 4))

    # Plot component FFTs
    for i in range(k):
        y0 = np.real(np.fft.fftshift(np.fft.fft(single_sinusoids[i])))
        plt.plot(ppm_val, y0, color="red", alpha=0.6, label="Components" if i == 0 else None)

    # Plot bootstrap peaks if provided
    if boot_samples is not None and omega_ppm is not None and show_bootstrap_peaks:
        idx_list = getoverlaps(boot_samples, omega_ppm, k)
        for i, idx in enumerate(idx_list):
            y_boot = np.real(np.fft.fftshift(np.fft.fft(single_sinusoids[idx])))
            peak_idx = np.argmax(y_boot)
            plt.scatter(ppm_val[peak_idx], y_boot[peak_idx], color="black", s=40,
                        label="Uncertain peaks" if i == 0 else None)

    # Plot model and data
    plt.plot(ppm_val, y_fft_cut, color="blue", label="Raw data")
    plt.plot(ppm_val, y_fft_hat, color="green", label="Fitted model")

    if show_residuals:
        plt.plot(ppm_val, y_fft_resi, color="orange", label="Residuals")

    # Formatting
    plt.xlim(xlim1_ppm, xlim2_ppm)
    plt.ylim(ylim1,ylim2)
    plt.xlabel('ppm')
    plt.ylabel('Intensity')
    plt.title(f"Model Results ({xlim1_ppm}-{xlim2_ppm} ppm)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()

    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{filename_prefix}.{save_format}"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, format=save_format, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()   
        